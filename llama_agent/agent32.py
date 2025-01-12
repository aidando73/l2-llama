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

# 512 is the default for fireworks on Llama-stack
# 4096 seems to be the max - https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct/discussions/6
MAX_OUTPUT_TOKENS = 512

PHASE1_ITERATIONS = 10

sampling_params = SamplingParams(
    strategy="greedy",
    max_tokens=MAX_OUTPUT_TOKENS,
)


tokenizer = Tokenizer.get_instance()
formatter = ChatFormat(tokenizer)

class Phase1PromptGenerator(PromptTemplateGeneratorBase):
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

            Your task is to locate the relevant file to the problem statement.

            You can perform only one of the following steps at a time:
            1. ANALYZE: 
            - Explain what you understand about the current state
            - Review the previous tool call (if any)
            - Describe the next steps that are needed to locate the relevant file

            2. EXECUTE:
            - Make the appropriate function call(s)
            - Format calls in the correct format specified below

            Locate the relevant file by making one or more function/tool calls.

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

            <|eot_id|>
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

PHASE1_TOOLS = [
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
        tool_name="pick_file",
        description=("Pick the file that is relevant to the problem statement."),
        parameters={
            "path": ToolParamDefinition(
                param_type="string",
                description="Path to file, e.g. `src/file.py` or `src/example/file.py`.",
                required=True,
            )
        },
    ),
]

def run_agent(
    client: LlamaStackClient,
    repo: str,
    problem_statement: str,
    sandbox_dir: Optional[str] = os.path.join(REPO_DIR, "sandbox"),
    eval_dir: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> Tuple[Literal["changes_made", "no_changes_made"], str, Optional[str]]:
    
    """
    PHASE 1: Locate the relevant file
    """
    print("PHASE 1 " + "-" * 80)
    message = Phase1PromptGenerator() \
        .gen(problem_statement=problem_statement, sandbox_dir=sandbox_dir, repo=repo, custom_tools=PHASE1_TOOLS) \
        .render()

    file_chosen = None
    for i in range(PHASE1_ITERATIONS):
        if file_chosen:
            break
        message += header("assistant")
        message += "ANALYSE:\n"
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

        print("ANALYSE:")
        print(magenta(analyse_statement))

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
        print(f"EXECUTE:\n{blue(raw_tool_call)}")
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
                result, result_msg = execute_phase_1_tool_call(tool_name, tool_params, sandbox_dir, repo)
            except Exception as e:
                result, result_msg = ("error", f"ERROR - Calling tool: {tool_name} {e}")

            message += f"Result: {result_msg}\n"

            if result == "success":
                # Truncate the result message to 200 characters since it can be long
                print("Result: " + result_msg[:200] + "...")
            else:
                print("Result: " + result_msg)

            message += f"<|eot_id|>"

            if result == "success" and tool_name == "pick_file":
                file_chosen = tool_params["path"]
                break
    
    if eval_dir:
        with open(
            os.path.join(eval_dir, "trajs", f"{instance_id}-prompt.txt"), "w"
        ) as f:
            f.write(message)
    else:
        with open("prompt.txt", "w") as f:
            f.write(message)

    """
    PHASE 2: Edit the file
    """
    # TODO


def header(role: Literal["user", "assistant", "system", "tool"]):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"

def token_count(message: str):
    return len(tokenizer.encode(message, bos=False, eos=False))

def chat_message(role: Literal["user", "assistant", "system", "tool"], content: str):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

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

def execute_phase_1_tool_call(
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

    elif tool_name == "pick_file":
        if (
            error := validate_param_exists("path", tool_params)
            or validate_not_symlink(sandbox_dir, repo, tool_params["path"])
            or validate_path_in_sandbox(sandbox_dir, repo, tool_params["path"])
            or validate_file_exists(sandbox_dir, repo, tool_params["path"])
            or validate_not_a_directory(sandbox_dir, repo, tool_params["path"])
        ):
            return ("error", error)
        return ("success", f"File picked - {tool_params['path']}")
    else:
        return ("error", f"ERROR - Unknown tool: {tool_name}")
    
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
