import os
from typing import Literal, Optional, Tuple, Union
import re
import json
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.shared_params.sampling_params import SamplingParams
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
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
# MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct-FP8"
ITERATIONS = 15

# 512 is the default for fireworks on Llama-stack
# 4096 seems to be the max - https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct/discussions/6
MAX_OUTPUT_TOKENS = 512

sampling_params = SamplingParams(
    strategy="greedy",
    max_tokens=MAX_OUTPUT_TOKENS,
)

tokenizer = Tokenizer.get_instance()
formatter = ChatFormat(tokenizer)


class L2SystemPromptGenerator(PromptTemplateGeneratorBase):
    def gen(
        self, problem_statement: str, sandbox_dir: str, repo: str, custom_tools: list[ToolDefinition]
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
            - Format calls in the correct format specified below

            Solve the users problem by making one or more function/tool calls.

            If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]

            Here is a list of functions in JSON format that you can invoke.

            [
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
                    "name": "{{tname}}",
                    "description": "{{tdesc}}",
                    "parameters": {
                        "type": "dict",
                        "required": {{ required_params | tojson }},
                        "properties": {
                            {%- for name, param in tparams.items() %}
                            "{{name}}": {
                                "type": "{{param.param_type}}",
                                "description": "{{param.description}}"{% if param.default %},
                                "default": "{{param.default}}"{% endif %}
                            }{% if not loop.last %},{% endif %}
                            {%- endfor %}
                        }
                    }
                }{% if not loop.last %},
                {% endif -%}
                {%- endfor %}
            ]

            Structure your response as:
            <|start_header_id|>assistant<|end_header_id|>

            ANALYZE:
            [Your analysis here]<|eot_id|>

            or:
            <|start_header_id|>assistant<|end_header_id|>

            EXECUTE:
            [Function call in the correct format specified above]<|eot_id|>

            Please start by listing out and viewing files in the repository to understand the problem.
            Then make the necessary changes to solve the problem.<|eot_id|>
            """
        )

        files_in_repo = "\n".join(
            list_files_in_repo(os.path.join(sandbox_dir, repo), depth=1)
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
    sandbox_dir: Optional[str] = os.path.join(REPO_DIR, "sandbox"),
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
        .gen(problem_statement=problem_statement, sandbox_dir=sandbox_dir, repo=repo, custom_tools=TOOLS)
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
            sampling_params=sampling_params,
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
        raw_tool_call = '['
        message += raw_tool_call
        print(f"Input tokens: {token_count(message)}")
        response = client.inference.completion(
            model_id=MODEL_ID,
            content=message,
            sampling_params=sampling_params,
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
                result, result_msg = execute_tool_call(tool_name, tool_params, sandbox_dir, repo)
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
        description="Edit a file. Specify the path to the file to edit.",
        parameters={
            "path": ToolParamDefinition(
                param_type="string",
                description="Path to file, e.g. `src/file.py` or `src/example/file.py`.",
                required=True,
            ),
            "old_str": ToolParamDefinition(
                param_type="string",
                description="The string in the file at `path` to replace. Must be non-empty.",
                required=True,
            ),
            "new_str": ToolParamDefinition(
                param_type="string",
                description="The new string to write to the file.",
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
        description=("If you have solved the problem, call this function to finish the task."
                      "Note that you must make changes to the codebase to finish the task, otherwise this function will fail."),
        parameters={},
    ),
]


def execute_tool_call(
    tool_name: str, tool_params: dict[str, str], sandbox_dir: str, repo: str
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
            or validate_not_symlink(sandbox_dir, repo, tool_params["path"])
            or validate_path_in_sandbox(sandbox_dir, repo, tool_params["path"])
            or validate_directory_exists(sandbox_dir, repo, tool_params["path"])
        ):
            return ("error", error)

        path = os.path.join(sandbox_dir, repo, tool_params["path"])
        files = list_files_in_repo(path, depth=1)
        return ("success", "\n".join(files))
    elif tool_name == "edit_file":
        if (
            error := validate_param_exists("path", tool_params)
            or validate_path_in_sandbox(sandbox_dir, repo, tool_params["path"])
            or validate_param_exists("new_str", tool_params)
            or validate_param_exists("old_str", tool_params)
            or validate_not_symlink(sandbox_dir, repo, tool_params["path"])
            or validate_file_exists(sandbox_dir, repo, tool_params["path"])
            or validate_not_a_directory(sandbox_dir, repo, tool_params["path"])
        ):
            return ("error", error)

        if tool_params["old_str"] == "":
            return ("error", "ERROR - old_str must be non-empty")

        path = os.path.join(sandbox_dir, repo, tool_params["path"])
        with open(f"{path}", "r") as f:
            file_content = f.read()
        with open(f"{path}", "w") as f:
            old_str = tool_params["old_str"]
            new_str = tool_params["new_str"]
            new_content = file_content.replace(old_str, new_str)
            f.write(new_content)
        diff = list(
            difflib.unified_diff(
                file_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
            )
        )
        if len(diff) == 0:
            return ("error", "ERROR - No changes made to file")
        else:
            return ("success", "File successfully updated\n" + "\n".join(diff))
    elif tool_name == "view_file":
        if (
            error := validate_param_exists("path", tool_params)
            or validate_not_symlink(sandbox_dir, repo, tool_params["path"])
            or validate_path_in_sandbox(sandbox_dir, repo, tool_params["path"])
            or validate_file_exists(sandbox_dir, repo, tool_params["path"])
            or validate_not_a_directory(sandbox_dir, repo, tool_params["path"])
        ):
            return ("error", error)

        path = os.path.join(sandbox_dir, repo, tool_params["path"])
        with open(f"{path}", "r") as f:
            file_content = f.read()
        return ("success", file_content)

    elif tool_name == "finish":
        return ("success", "Task finished")
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
    if not is_valid_python_list(content):
        content = content.strip()

        # Add square brackets if missing
        if not content.startswith("["):
            content = f"[{content}"
        if not content.endswith("]"):
            content = f"{content}]"

    try:
        result = parse_python_list_for_function_calls(content)
        if is_valid_python_list(content):
            # Add the original tool content to each result tuple
            result = [(name, params) for name, params in result]
            return result
        else:
            return [(
                "error",
                "Tool call invalid syntax: " + content,
            )]
    except Exception as e:
        return [(
            "error",
            "Tool call invalid syntax: Could not parse tool call: "
            + content
            + " "
            + str(e),
        )]

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


def validate_path_in_sandbox(sandbox_dir: str, repo: str, path: str) -> Optional[str]:
    """
    Validate that a path stays within the sandbox directory.

    Args:
        path (str): The path to validate

    Returns:
        Optional[str]: Error message if path is invalid, None if valid
    """
    # Resolve the absolute path after translation to catch any ../ tricks
    resolved_path = os.path.abspath(os.path.join(sandbox_dir, repo, path))
    sandbox_path = os.path.abspath(sandbox_dir)

    if not resolved_path.startswith(sandbox_path):
        # From the agent's perspective, any paths not in the sandbox don't exist
        return f"ERROR - File {path} does not exist"
    return None


def validate_not_symlink(sandbox_dir: str, repo: str, path: str) -> Optional[str]:
    resolved_path = os.path.abspath(os.path.join(sandbox_dir, repo, path))
    if os.path.islink(resolved_path):
        return f"ERROR - File {path} is a symlink. Simlinks not allowed"
    return None


def validate_file_exists(sandbox_dir: str, repo: str, path: str) -> Optional[str]:
    resolved_path = os.path.abspath(os.path.join(sandbox_dir, repo, path))
    if not os.path.exists(resolved_path):
        return f"ERROR - File {path} does not exist. Please ensure the file exists."
    return None


def validate_not_a_directory(sandbox_dir: str, repo: str, path: str) -> Optional[str]:
    resolved_path = os.path.abspath(os.path.join(sandbox_dir, repo, path))
    if os.path.isdir(resolved_path):
        return f"ERROR - File {path} is a directory. Please ensure the path references a file, not a directory."
    return None


def validate_directory_exists(sandbox_dir: str, repo: str, path: str) -> Optional[str]:
    resolved_path = os.path.abspath(os.path.join(sandbox_dir, repo, path))
    if not os.path.exists(resolved_path):
        return f"ERROR - Directory {path} does not exist. Please ensure the directory exists."
    return None


def chat_message(role: Literal["user", "assistant", "system", "tool"], content: str):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"


def header(role: Literal["user", "assistant", "system", "tool"]):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"


def token_count(message: str):
    return len(tokenizer.encode(message, bos=False, eos=False))


# Very hacky implementation
# But we'll see how well it works
def replace_content(old_file_content: str, old_content: str, new_content: str):
    """
    Replaces old_content with new_content in old_file_content.
    Does so without worring about identation
    We count the indentation of old_file_content
    Remove all leading whitespace from old_content and new_content
    And then add it back in at the end
    """
    if dedent(old_content) == dedent(new_content):
        raise AssertionError("Old content and new content are identical. File unchanged.")

    from math import inf
    def get_common_indentation(lines: list[str]) -> str:
        """
        Returns:
            Tuple[int, Optional[str]]:
                - indent_char (Optional[str]): The character type of the indentation, either " " or "\t" or '' if no whitespace
                - common_indentation (int): The common indentation of the lines
        """
        common_indentation = inf
        indent_char = None
        for line in lines:
            # Assume tabs or spaces only
            if whitespace := re.match(r'^[ \t]+', line):
                if indent_char is None:
                    # Just get the first whitespace character we encounter
                    indent_char = whitespace.group(0)[0]
                if len(whitespace.group(0)) < common_indentation:
                    common_indentation = len(whitespace.group(0))
            else:
                # If we encounter a line without whitespace, then there's no whitespace to remove
                return ('', 0)
        return indent_char, common_indentation

    old_file_content_lines = old_file_content.splitlines(keepends=True)
    old_content_lines = old_content.splitlines(keepends=True)

    if len(old_content_lines) > len(old_file_content_lines):
        return old_file_content

    m = len(old_content_lines)
    for i in range(len(old_file_content_lines) - m + 1):
        lines = old_file_content_lines[i:i + m]
        indent_char, common_indentation = get_common_indentation(lines)

        # Check if the old content is in the dedented content
        content_dedented = dedent("".join(lines))
        old_content_dedented = dedent(old_content)
        if old_content_dedented in content_dedented:
            new_file_dedented = dedent(new_content)
            content_dedented = content_dedented.replace(old_content_dedented, new_file_dedented)
            content_dedented = content_dedented.splitlines(keepends=True)
            res = ""
            for line in content_dedented:
                res += indent_char * common_indentation + line
            return "".join(old_file_content_lines[:i]) + res + "".join(old_file_content_lines[i + m:])

    raise AssertionError("Old content not found in file")