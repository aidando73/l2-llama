import json
import os
from typing import Literal, Optional, Tuple, Union
from llama_stack_client import LlamaStackClient
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.api.datatypes import StopReason
from llama_models.llama3.api.tool_utils import (
    is_valid_python_list,
    parse_python_list_for_function_calls,
)
import re
from file_tree_5 import list_files_in_repo
from __init__ import REPO_DIR
from ansi import red, green

# Currently only supports 3.3-70B-Instruct at the moment since it depends on the 3.3/3.2 tool prompt format
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
ITERATIONS = 2

SANDBOX_DIR = os.path.join(REPO_DIR, "sandbox")
# We give the agent a virtual working directory so it doesn't have to worry about long absolute paths
AGENT_WORKING_DIR = "/workspace/"

formatter = ChatFormat(Tokenizer.get_instance())

def run_agent(
    client: LlamaStackClient, repo: str, issue_title: str, issue_body: str
) -> Tuple[Literal["changes_made", "no_changes_made"], str, Optional[str]]:
    """
    Returns:
        Tuple[Literal["changes_made", "no_changes_made"], str, Optional[str]]:
            ("changes_made", pr_title, pr_body): "changes_made", the PR title, and the PR body
            or ("no_changes_made", reasoning, None): "no_changes_made", the reason why no changes were made, and None
    """

    message = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an expert software engineer.
    You will be given a problem statement in <problem_statement>

    Based on the <problem_statement>, you will need to make one or more function/tool calls to achieve the purpose.
    If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
    also point it out. You should only return the function call in tools call sections.

    If you decide to invoke any of the function(s), you MUST put it in the format of <tool>[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]</tool>
    If you decide to invoke multiple functions, you MUST put commas between the function calls. E.g., <tool>[func_name1(params), func_name2(params), func_name3(params)]</tool>

    Here is a list of functions in JSON format that you can invoke.

    [
        {
            "name": "list_files",
            "description": "List all files in a directory.",
            "parameters": {
                "type": "dict",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to a directory, e.g. `/workspace/django`. If referencing a file, will return the name of the file."
                    }
                },
            }
        },
        {
            "name": "edit_file",
            "description": "Edit a file. Specify the path to the file and the new_str to write to it. If old_str is specified, only the old_str will be replaced with new_str, otherwise the entire file will be replaced by new_str.",
            "parameters": {
                "type": "dict",
                "required": ["path", "new_str"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory, e.g. `/workspace/django/file.py` or `/workspace/django`."
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The string in the file at `path` to replace. If not specified, the entire file will be replaced by new_str"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The new string to write to the file. If the old_str is specified, only the old_str will be replaced with new_str, otherwise the entire file will be replaced by new_str."
                    }
                }
            }
        },
        {
            "name": "view_file",
            "description": "View a file",
            "parameters": {
                "type": "dict",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute path to the file to view, e.g. `/workspace/django/file.py` or `/workspace/django`."
                    }
                }
            }
        },
        {
            "name": "finish",
            "description": "If you have solved the problem, you can call this function to finish the task.",
            "parameters": {}
        }
    ]

    Please explain your reasoning before you make any edits in a <thinking> tag.

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    <working_directory>
    %working_directory%
    </working_directory>

    <file_tree>
    %file_tree%
    </file_tree>

    <problem_statement>
    %problem_statement%
    </problem_statement>

    You are in the working directory as specified in <working_directory>. Please specify paths in absolute paths only.
    I have included the top level files and directories in the repository in <file_tree>.
    Please start by listing out and viewing files in the repository to understand the problem.<|eot_id|>
    """.strip()

    problem_statement = (
        "Issue title: " + issue_title + "\n" + "Issue body: " + issue_body
    )

    message = message.replace("%working_directory%", os.path.join(SANDBOX_DIR, {repo}))
    message = message.replace(
        "%file_tree%", "\n".join(list_files_in_repo(os.path.join(SANDBOX_DIR, repo), depth=2))
    )
    message = message.replace("%problem_statement%", problem_statement)

    finished = False

    for i in range(ITERATIONS):
        print(f"Iteration {i+1} of {ITERATIONS}")
        if finished:
            break
        message += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        response = client.inference.completion(
            model_id=MODEL_ID,
            content=message,
        )
        thinking_match = re.search(
            r"<thinking>(.*?)</thinking>", response.content, re.DOTALL
        )
        if thinking_match:
            print("\033[94mThinking:", thinking_match.group(1).strip(), "\033[0m")
        else:
            # Check for any text outside of tool tags
            non_tool_content = re.sub(
                r"<tool>.*?</tool>", "", response.content, flags=re.DOTALL
            ).strip()
            if non_tool_content:
                print(f"\033[94mThinking: {non_tool_content}\033[0m")

        message += response.content
        message += f"<|eot_id|>"

        # Parse tool tags from response
        tool_calls = parse_tool_calls(response.content)
        for tool_call in tool_calls:
            if tool_call[0] == "error":
                _, error_message = tool_call
                msg = f"ERROR - Could not parse tool call: {error_message}"
                print(red(msg))
                message += f"<|start_header_id|>tool<|end_header_id|>\n\n"
                message += f"{msg}\n"
                message += f"<|eot_id|>"
                continue

            tool_name, tool_params = tool_call
            msg = f"Executing tool call: [{tool_name}{display_tool_params(tool_params)}]"
            message += f"<|start_header_id|>tool<|end_header_id|>\n\n"
            message += msg + "\n"
            print(green(msg))
            try:
                if tool_name == "list_files":
                    if "path" not in tool_params:
                        msg = f"Result: ERROR - path not found in tool params: {display_tool_params(tool_params)}"
                        print(red(msg))
                        message += f"{msg}\n"
                        continue

                    path = tool_params["path"]

                    try:
                        files = list_files_in_repo(path, depth=1)
                        message += "Result:\n"
                        message += "\n".join(files)
                    except FileNotFoundError as e:
                        msg = f"Result: ERROR - Directory not found: {e}"
                        print(red(msg))
                        message += f"{msg}\n"
                        continue
                elif tool_name == "edit_file":
                    if "new_str" not in tool_params:
                        msg = f"Result: ERROR - new_str not found in tool params: {display_tool_params(tool_params)}"
                        print(red(msg))
                        message += f"{msg}\n"
                        continue
                    try:
                        path = translate_path(tool_params["path"])

                        if "old_str" in tool_params:
                            with open(f"{path}", "r") as f:
                                file_content = f.read()
                            with open(f"{path}", "w") as f:
                                old_str = tool_params["old_str"]
                                new_str = tool_params["new_str"]
                                new_content = file_content.replace(old_str, new_str)
                                f.write(new_content)
                        else:
                            with open(f"{path}", "w") as f:
                                f.write(tool_params["new_str"])
                        message += f"Result: File successfully updated\n"
                    except FileNotFoundError:
                        msg = f"Result: ERROR - File {tool_params['path']} not found. Please ensure the path is an absolute path and that the file exists."
                        print(red(msg))
                        message += f"{msg}\n"
                    except IsADirectoryError:
                        msg = f"Result: ERROR - Path {tool_params['path']} is a directory. Please ensure the path references a file, not a directory."
                        print(red(msg))
                        message += f"{msg}\n"
                elif tool_name == "view_file":
                    try:
                        path = translate_path(tool_params["path"])
                        with open(f"{path}", "r") as f:
                            file_content = f.read()
                        message += f"Result: {file_content}\n"
                    except FileNotFoundError:
                        msg = f"Result: ERROR - File {tool_params['path']} not found. Please ensure the path is an absolute path and that the file exists."
                        print(red(msg))
                        message += f"{msg}\n"
                    except IsADirectoryError:
                        msg = f"Path {tool_params['path']} is a directory. Please ensure the path references a file, not a directory."
                        print(red(msg))
                        message += f"{msg}\n"
                elif tool_name == "finish":
                    finished = True
                    message += f"Result: Task marked as finished\n"
                else:
                    msg = f"Result: ERROR - Unknown tool: {tool_name}"
                    print(red(msg))
                    message += f"{msg}\n"
            except Exception as e:
                msg = f"Result: ERROR - Calling tool: {tool_name} {e}"
                print(red(msg))
                message += f"{msg}\n"
            message += f"<|eot_id|>"

    if finished:
        print(green("Agent marked as finished"))
    else:
        print(red("Max iterations reached"))

    # Create a PR title
    message += f"<|start_header_id|>user<|end_header_id|>\n\n"
    message += "Please create a PR title that summarizes the changes you've made. Do not include any leading or trailing punctuation."
    message += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    response = client.inference.completion(
        model_id=MODEL_ID,
        content=message,
    )
    pr_title = response.content

    diff = os.popen(f"cd sandbox/{repo} && git diff").read()

    if not diff:
        print(f"No changes were made - agent explaining why...")
        message += f"<|start_header_id|>user<|end_header_id|>\n\n"
        message += (
            "No changes were made."
            "Could you explain your reasoning for not making any changes?"
            "Please write it in GitHub Flavored Markdown."
            "Also provide some next steps to fix the issue."
        )
        message += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        response = client.inference.completion(
            model_id=MODEL_ID,
            content=message,
        )
        reasoning = response.content

        return ("no_changes_made", reasoning, None)

    # Create a PR body
    message += f"<|start_header_id|>user<|end_header_id|>\n\n"
    message += (
        "Summarizing all of the changes and thinking you've done,"
        "please write a PR body that explains the changes you've made."
        "Please write it in GitHub Flavored Markdown."
    )
    message += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # Llama sometimes includes an unnecessary "#PR body" title so we add it here to make sure it's not included
    message += "## PR Body\n\n"

    response = client.inference.completion(
        model_id=MODEL_ID,
        content=message,
    )
    pr_body = response.content

    return "changes_made", pr_title, pr_body



def translate_path(path: str) -> str:
    if path.startswith(AGENT_WORKING_DIR):
        return os.path.join(SANDBOX_DIR, path[len(AGENT_WORKING_DIR) :])
    else:
        return os.path.join(SANDBOX_DIR, path)

def parse_tool_calls(
    content,
) -> list[Union[tuple[str, dict[str, str]], tuple[Literal["error"], str]]]:
    """
    Parse tool calls from the content.

    Args:
        content (str): The content to parse tool calls from.

    Returns:
        list[tuple]: A list of tuples containing:
            - name (str): The name of the tool
            - params (dict): The parameters of the tool
            or ("error", error_message) if the tool call is invalid
    """
    tool_calls = []
    for match in re.finditer(r"<tool>(.*?)</tool>", content, re.DOTALL):
        tool_content = match.group(1)
        if not is_valid_python_list(tool_content):
            tool_content = tool_content.strip()

            # Add square brackets if missing
            if not tool_content.startswith("["):
                tool_content = f"[{tool_content}"
            if not tool_content.endswith("]"):
                tool_content = f"{tool_content}]"

        try:
            result = parse_python_list_for_function_calls(tool_content)
            if is_valid_python_list(tool_content):
                # Add the original tool content to each result tuple
                result = [(name, params) for name, params in result]
                tool_calls.extend(result)
            else:
                tool_calls.append(
                    (
                        "error",
                        "Tool call invalid syntax: " + match.group(0),
                    )
                )
        except Exception as e:
            tool_calls.append(
                (
                    "error",
                    "Tool call invalid syntax: Could not parse tool call: "
                    + match.group(0)
                    + " "
                    + str(e),
                )
            )

    return tool_calls

def display_tool_params(tool_params: dict[str, str]):
    return (
        "("
        + ", ".join(
            [
                param_name + '="' + str(param_value) + '"'
                for param_name, param_value in tool_params.items()
            ]
        )
        + ")"
    )
