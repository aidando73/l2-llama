import os
from typing import Literal, Optional, Tuple, Union
import re
import json
from llama_stack_client import LlamaStackClient
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.api.datatypes import StopReason
from llama_models.llama3.api.tool_utils import (
    is_valid_python_list,
    parse_python_list_for_function_calls,
)
from llama_models.llama3.api.datatypes import (
    ToolDefinition,
    ToolParamDefinition,
)
from llama_models.llama3.prompt_templates.system_prompts import (
    FunctionTagCustomToolGenerator,
    PromptTemplateGeneratorBase,
    PromptTemplate,
)
from llama_agent.utils.file_tree import list_files_in_repo
from llama_agent import REPO_DIR
from llama_agent.utils.ansi import red, yellow, magenta, blue, cyan
from subprocess import run
from textwrap import dedent
import textwrap
import difflib

# Currently only supports 3.3-70B-Instruct at the moment since it depends on the 3.3/3.2 tool prompt format
# MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct-FP8"
ITERATIONS = 15

SANDBOX_DIR = os.path.join(REPO_DIR, "sandbox")
tokenizer = Tokenizer.get_instance()
formatter = ChatFormat(tokenizer)


class L2SystemPromptGenerator(PromptTemplateGeneratorBase):
    def gen(
        self, problem_statement: str, repo: str, custom_tools: list[ToolDefinition]
    ) -> str:
        template_str = textwrap.dedent(
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            
            You are an expert software engineer. You are given the following problem:
            <problem_statement>
            {{ problem_statement }}
            </problem_statement>

            The repo is called {{ repo }}.

            Here is the file tree of the repository:
            <file_tree>
            {{ file_tree }}
            </file_tree>

            Your task is to solve the user's problem through analysis and appropriate function/tool calls.

            You can perform only one of the following steps at a time:
            1. ANALYZE: 
            - Explain what you understand about the current state
            - Review the previous tool call (if any)
            - Describe the next steps that are needed to solve the problem

            2. EXECUTE:
            - Make the appropriate function call(s)
            - Format calls in proper JSON

            Solve the users problem by making one or more function/tool calls.
            Here is a list of functions in JSON format:
            {% for t in custom_tools -%}
            {# manually setting up JSON because jinja sorts keys in unexpected ways -#}
            {%- set tname = t.tool_name -%}
            {%- set tdesc = t.description -%}
            {%- set tparams = t.parameters -%}
            {%- set required_params = [] -%}
            {%- for name, param in tparams.items() if param.required == true -%}
                {%- set _ = required_params.append(name) -%}
            {%- endfor -%}
            {
                "type": "function",
                "function": {
                    "name": "{{tname}}",
                    "description": "{{tdesc}}",
                    "parameters": {
                        "type": "object",
                        "properties": [
                            {%- for name, param in tparams.items() %}
                            {
                                "{{name}}": {
                                    "type": "object",
                                    "description": "{{param.description}}"
                                }
                            }{% if not loop.last %},{% endif %}
                            {%- endfor %}
                        ],
                        "required": {{ required_params | tojson }}
                    }
                }
            }
            {% endfor %}
            Return function calls in JSON format.

            Structure your response as:
            <|start_header_id|>assistant<|end_header_id|>

            ANALYZE:
            [Your analysis here]<|eot_id|>

            or:
            <|start_header_id|>assistant<|end_header_id|>

            EXECUTE:
            [Function call in JSON format]<|eot_id|>

            Please start by listing out and viewing files in the repository to understand the problem.
            Then make the necessary changes to solve the problem.<|eot_id|>
            """
        )

        files_in_repo = "\n".join(
            list_files_in_repo(os.path.join(SANDBOX_DIR, repo), depth=1)
        )
        return PromptTemplate(
            template_str.lstrip("\n"),
            {
                "custom_tools": [t.model_dump() for t in custom_tools],
                "problem_statement": problem_statement,
                "repo": repo,
                "file_tree": files_in_repo,
            },
        )


def run_agent(
    client: LlamaStackClient,
    repo: str,
    problem_statement: str,
    eval_dir: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> Tuple[Literal["changes_made", "no_changes_made"], str, Optional[str]]:
    """
    Returns:
        Tuple[Literal["changes_made", "no_changes_made"], str, Optional[str]]:
            ("changes_made", pr_title, pr_body): "changes_made", the PR title, and the PR body
            or ("no_changes_made", reasoning, None): "no_changes_made", the reason why no changes were made, and None
    """

    message = (
        L2SystemPromptGenerator()
        .gen(problem_statement=problem_statement, repo=repo, custom_tools=TOOLS)
        .render()
    )

    finished = False
    for i in range(ITERATIONS):
        print("\n")
        print(f"Iteration {i+1} of {ITERATIONS}")
        print("-" * 80)

        if finished:
            break

        # ANALYSE
        message += header("assistant")
        message += "ANALYSE: \n"
        print(f"Input tokens: {token_count(message)}")
        response = client.inference.completion(
            model_id=MODEL_ID,
            content=message,
        )
        if "EXECUTE:" in response.content:
            # Sometimes the agent will respond with the EXECUTE statement
            # we want it to respond in separate turns so it's easier to pre-empt the model
            # and parse the tool call
            # print("DEBUG", response.content)
            analyse_statement = response.content[: response.content.find("EXECUTE:")]
            analyse_statement = analyse_statement.rstrip()
        else:
            analyse_statement = response.content
        message += analyse_statement
        message += f"<|eot_id|>"

        print(f"ANALYSE: {magenta(analyse_statement)}")

        # EXECUTE
        message += header("assistant")
        message += "EXECUTE: \n"
        # Pre-empt the tool call to prevent poor tool call formatting
        raw_tool_call = '[{"type": "function", "name": "'
        message += raw_tool_call
        print(f"Input tokens: {token_count(message)}")
        response = client.inference.completion(
            model_id=MODEL_ID,
            content=message,
        )
        message += response.content
        message += f"<|eot_id|>"

        raw_tool_call += response.content
        print(f"EXECUTE: {blue(raw_tool_call)}")
        # Evaluate tool calls
        tool_calls = parse_tool_calls(raw_tool_call)
        for tool_call in tool_calls:

            if tool_call[0] == "error":
                _, error_message = tool_call
                msg = f"ERROR - Could not parse tool call: {error_message}"
                print(red(msg))
                message += chat_message("tool", msg)
                continue

            tool_name, tool_params = tool_call
            msg = f"[{tool_name}{display_tool_params(tool_params)}]"
            message += header("tool")
            message += "Executing tool call: " + msg + "\n"
            print("Executing tool call: " + cyan(msg))

            try:
                # Custom edit_file tool call
                # We don't use the tool call format for old and new content becuase Llama struggles with it
                # Llama needs to escape newlines and double quotes, it's easier to just prompt for the old and new content
                if tool_name == "edit_file":
                    if (error := validate_param_exists("path", tool_params)
                        or validate_path_in_sandbox(repo, tool_params["path"])
                        or validate_not_symlink(repo, tool_params["path"])
                        or validate_file_exists(repo, tool_params["path"])
                        or validate_not_a_directory(repo, tool_params["path"])
                    ):
                        result, result_msg = ("error", error)
                    else:
                        path = os.path.join(SANDBOX_DIR, repo, tool_params["path"])

                        # Prompt for old content
                        message += chat_message("tool", "Please provide the old content to replace. Please format it in ```\nCODE\n``` format.")
                        message += "<|eot_id|>"
                        print(f"Input tokens: {token_count(message)}")
                        print("OLD_CONTENT: ")
                        message += header("assistant")
                        response = client.inference.completion(
                            model_id=MODEL_ID,
                            content=message,
                        )
                        message += response.content
                        old_content = strip_code_block(response.content)
                        print(blue(old_content))
                        message += "<|eot_id|>"
                        message += header("assistant")

                        # Prompt for new content
                        message += chat_message("tool", "Please provide the new content to replace the old content with. Please format it in ```\nCODE\n``` format.")
                        message += "<|eot_id|>"
                        message += header("assistant")
                        print(f"Input tokens: {token_count(message)}")
                        print("NEW_CONTENT: ")
                        response = client.inference.completion(
                            model_id=MODEL_ID,
                            content=message,
                        )
                        message += response.content
                        new_content = strip_code_block(response.content)
                        print(blue(new_content))
                        message += "<|eot_id|>"

                        with open(path, "r") as f:
                            old_file_content = f.read()
    
                        new_file_content = old_file_content.replace(old_content, new_content)

                        with open(path, "w") as f:
                            f.write(new_file_content)
                        # Get diff between old and new content
                        diff = list(
                            difflib.unified_diff(
                                old_file_content.splitlines(keepends=True),
                                new_file_content.splitlines(keepends=True),
                                fromfile="before",
                                tofile="after",
                            )
                        )
                        if len(diff) == 0:
                            result, result_msg = ("error", "ERROR - No changes made to file")
                        else:
                            result, result_msg = ("success", "File successfully updated\n" + "\n".join(diff))
                else:
                    result, result_msg = execute_tool_call(tool_name, tool_params, repo)
            except Exception as e:
                result, result_msg = ("error", f"ERROR - Calling tool: {tool_name} {e}")

            message += f"Result: {result_msg}\n"

            if result == "success":
                # Truncate the result message to 200 characters since it can be long
                print("Result: " + result_msg[:200] + "...")
            else:
                print("Result: " + result_msg)

            message += f"<|eot_id|>"

            if result == "success" and tool_name == "finish":
                finished = True

    if finished:
        print(blue("Agent marked as finished"))
    else:
        print(yellow("Max iterations reached"))

    if eval_dir:
        with open(
            os.path.join(eval_dir, "trajs", f"{instance_id}-prompt.txt"), "w"
        ) as f:
            f.write(message)
    else:
        with open("prompt.txt", "w") as f:
            f.write(message)


def strip_code_block(content: str) -> str:
    # Strip out leading backticks if any
    content = content.strip()
    if "```" not in content:
        return content
    
    # Strip out leading backticks
    backticks = re.search(r"```.*\n", content)
    if backticks:
        content = content[backticks.end():]

    # Strip out trailing backticks
    backticks = re.search(r"\n```", content)
    if backticks:
        content = content[:backticks.start()]

    return content

TOOLS = [
    ToolDefinition(
        tool_name="list_files",
        description="List all files in a directory.",
        parameters={
            "path": ToolParamDefinition(
                param_type="string",
                description="Path to a directory. E.g., `src/` or `src/example` If referencing a file, will return the name of the file.",
                required=True,
            )
        },
    ),
    ToolDefinition(
        tool_name="edit_file",
        description="Edit a file. Specify the path to the file to edit. You will be prompted for the old and new content to edit the file.",
        parameters={
            "path": ToolParamDefinition(
                param_type="string",
                description="Path to file, e.g. `src/file.py` or `src/example/file.py`.",
                required=True,
            ),
        },
    ),
    ToolDefinition(
        tool_name="view_file",
        description="View a file",
        parameters={
            "path": ToolParamDefinition(
                param_type="string",
                description="Path to file, e.g. `src/file.py` or `src/example/file.py`.",
                required=True,
            )
        },
    ),
    ToolDefinition(
        tool_name="finish",
        description="If you have solved the problem, you can call this function to finish the task.",
        parameters={},
    ),
]


def execute_tool_call(
    tool_name: str, tool_params: dict[str, str], repo: str
) -> Union[Tuple[Literal["success"], str], Tuple[Literal["error"], str]]:
    """
    Execute a tool call and return a message indicating the result of the tool call.

    Args:
        tool_name (str): The name of the tool to execute.
        tool_params (dict[str, str]): The parameters to pass to the tool.

    Returns:
        Union[Tuple[Literal["success"], str], Tuple[Literal["error"], str]]:
            ("success", result): The result of the tool call.
            ("error", error_message): The error message if the tool call failed.
    """
    if tool_name == "list_files":
        if (
            error := validate_param_exists("path", tool_params)
            or validate_not_symlink(repo, tool_params["path"])
            or validate_path_in_sandbox(repo, tool_params["path"])
            or validate_directory_exists(repo, tool_params["path"])
        ):
            return ("error", error)

        path = os.path.join(SANDBOX_DIR, repo, tool_params["path"])
        files = list_files_in_repo(path, depth=1)
        return ("success", "\n".join(files))

    elif tool_name == "view_file":
        if (
            error := validate_param_exists("path", tool_params)
            or validate_not_symlink(repo, tool_params["path"])
            or validate_path_in_sandbox(repo, tool_params["path"])
            or validate_file_exists(repo, tool_params["path"])
            or validate_not_a_directory(repo, tool_params["path"])
        ):
            return ("error", error)

        path = os.path.join(SANDBOX_DIR, repo, tool_params["path"])
        with open(f"{path}", "r") as f:
            file_content = f.read()
        return ("success", file_content)

    elif tool_name == "finish":
        return ("success", "Task marked as finished")

    else:
        return ("error", f"ERROR - Unknown tool: {tool_name}")


def parse_tool_calls(
    content,
) -> list[Union[tuple[str, dict[str, str]], tuple[Literal["error"], str]]]:
    """
    Parse tool calls from the content.

    Args:
        content (str): The content to parse tool calls from.

    Returns:
        list[Union[tuple[str, dict[str, str]], tuple[Literal["error"], str]]: Either:
            tuple[str, dict[str, str]]:
                - name (str): The name of the tool
                - params (dict): The parameters of the tool
            or tuple[Literal["error"], str] if the tool call could not be parsed:
                - "error"
                - error_message (str): The error message
    """
    tool_calls = []
    for match in re.finditer(r"<tool>(.*?)</tool>", content, re.DOTALL):
        raw_function = match.group(1)
        try:
            function = json.loads(raw_function)
            if "type" not in function or function["type"] != "function":
                return (
                    "error",
                    "Tool call invalid syntax: "
                    + raw_function
                    + 'expected <tool>{"type": "function", ...}</tool>',
                )
            if "name" not in function:
                return (
                    "error",
                    "Tool call invalid syntax: "
                    + raw_function
                    + 'expected <tool>{"type": "function", "name": "func_name", ...}</tool>',
                )
            function_name = function["name"]
            args = function["parameters"]
            tool_calls.append((function_name, args))
        except Exception as e:
            tool_calls.append(
                ("error", f"Tool call invalid syntax: {raw_function} {e}")
            )

    print(len(tool_calls), is_json(content))
    if len(tool_calls) == 0 and is_json(content):
        # Sometimes the tool call is a list of functions
        function = json.loads(content)
        if isinstance(function, list):
            for func in function:
                if "type" not in func or func["type"] != "function":
                    tool_calls.append(
                        (
                            "error",
                            "Tool call invalid syntax: "
                            + content
                            + 'expected {"type": "function", ...}',
                        )
                    )
                if "name" not in func:
                    tool_calls.append(
                        (
                            "error",
                            "Tool call invalid syntax: "
                            + content
                            + 'expected {"type": "function", "name": "func_name", ...}',
                        )
                    )
                tool_calls.append((func["name"], func["parameters"]))
        else:
            if "type" not in function or function["type"] != "function":
                tool_calls.append(
                    (
                        "error",
                        "Tool call invalid syntax: "
                        + content
                        + 'expected {"type": "function", ...}',
                    )
                )
            if "name" not in function:
                tool_calls.append(
                    (
                        "error",
                        "Tool call invalid syntax: "
                        + content
                        + 'expected {"type": "function", "name": "func_name", ...}',
                    )
                )
            tool_calls.append((function["name"], function["parameters"]))
    if len(tool_calls) == 0:
        return [("error", content)]
    return tool_calls


def is_json(s):
    try:
        parsed = json.loads(s)
        # Return True for valid objects and not for ints, strings, etc
        return isinstance(parsed, dict) or isinstance(parsed, list)
    except json.JSONDecodeError:
        return False
    return True


CUSTOM_TOOL_CALL_PATTERN = r"<function=(?P<function_name>[^}]+)>(?P<args>{.*?})"


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


def validate_param_exists(
    param_name: str, tool_params: dict[str, str]
) -> Optional[str]:
    if param_name not in tool_params:
        return f"ERROR - {param_name} not found in tool params: {display_tool_params(tool_params)}"
    return None


def validate_path_in_sandbox(repo: str, path: str) -> Optional[str]:
    """
    Validate that a path stays within the sandbox directory.

    Args:
        path (str): The path to validate

    Returns:
        Optional[str]: Error message if path is invalid, None if valid
    """
    # Resolve the absolute path after translation to catch any ../ tricks
    resolved_path = os.path.abspath(os.path.join(SANDBOX_DIR, repo, path))
    sandbox_path = os.path.abspath(SANDBOX_DIR)

    if not resolved_path.startswith(sandbox_path):
        # From the agent's perspective, any paths not in the sandbox don't exist
        return f"ERROR - File {path} does not exist"
    return None


def validate_not_symlink(repo: str, path: str) -> Optional[str]:
    resolved_path = os.path.abspath(os.path.join(SANDBOX_DIR, repo, path))
    if os.path.islink(resolved_path):
        return f"ERROR - File {path} is a symlink. Simlinks not allowed"
    return None


def validate_file_exists(repo: str, path: str) -> Optional[str]:
    resolved_path = os.path.abspath(os.path.join(SANDBOX_DIR, repo, path))
    if not os.path.exists(resolved_path):
        return f"ERROR - File {path} does not exist. Please ensure the file exists."
    return None


def validate_not_a_directory(repo: str, path: str) -> Optional[str]:
    resolved_path = os.path.abspath(os.path.join(SANDBOX_DIR, repo, path))
    if os.path.isdir(resolved_path):
        return f"ERROR - File {path} is a directory. Please ensure the path references a file, not a directory."
    return None


def validate_directory_exists(repo: str, path: str) -> Optional[str]:
    resolved_path = os.path.abspath(os.path.join(SANDBOX_DIR, repo, path))
    if not os.path.exists(resolved_path):
        return f"ERROR - Directory {path} does not exist. Please ensure the directory exists."
    return None


def chat_message(role: Literal["user", "assistant", "system", "tool"], content: str):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"


def header(role: Literal["user", "assistant", "system", "tool"]):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"


def token_count(message: str):
    return len(tokenizer.encode(message, bos=False, eos=False))

