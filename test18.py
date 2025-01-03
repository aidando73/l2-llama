from app19 import Issue
import pytest
from subprocess import run


class TestIssue:
    def test_basic_url(self):
        url = "https://github.com/aidando73/bitbucket-syntax-highlighting/issues/67"
        issue = Issue(url)
        assert issue.owner == "aidando73"
        assert issue.repo == "bitbucket-syntax-highlighting"
        assert issue.issue_number == 67

    def test_basic_url_without_https(self):
        url = "github.com/aidando73/bitbucket-syntax-highlighting/issues/67"
        issue = Issue(url)
        assert issue.owner == "aidando73"
        assert issue.repo == "bitbucket-syntax-highlighting"
        assert issue.issue_number == 67

    def test_invalid_url(self):
        with pytest.raises(ValueError, match="Expected github.com as the domain"):
            Issue("https://not-github.com/owner/repo/issues/1")

    def test_invalid_url_no_issue(self):
        with pytest.raises(ValueError, match="Invalid GitHub issue URL format"):
            Issue("https://github.com/owner/repo")

    def test_invalid_url_no_issue_number(self):
        with pytest.raises(ValueError, match="Expected an issue number in the URL"):
            Issue("https://github.com/owner/repo/issues/")

    def test_issue_number_is_not_integer(self):
        with pytest.raises(ValueError, match="Expected an integer issue number"):
            Issue("https://github.com/owner/repo/issues/not_an_integer")


from agent17_5 import display_tool_params


class TestDisplayToolParams:
    def test_no_params(self):
        assert display_tool_params({}) == "()"

    def test_one_param(self):
        assert display_tool_params({"a": "b"}) == '(a="b")'

    def test_three_params(self):
        assert (
            display_tool_params({"a": "b", "c": "d", "e": "f"})
            == '(a="b", c="d", e="f")'
        )


from agent17_5 import parse_tool_calls


class TestParseToolCallFromContent:
    def test_basic_tool_call(self):
        content = '<tool>[func1(a="1", b="2")]</tool>'
        assert parse_tool_calls(content) == [("func1", {"a": "1", "b": "2"})]

    def test_empty_arg(self):
        content = '<tool>[func1(a="1", b=)]</tool>'

        res = parse_tool_calls(content)

        assert len(res) == 1
        error, error_message = res[0]
        assert error == "error"
        assert "Tool call invalid syntax" in error_message

    def test_handles_missing_left_matching_bracket(self):
        content = "<tool>func1()]</tool>"

        res = parse_tool_calls(content)

        assert len(res) == 1
        tool_name, tool_params = res[0]
        assert tool_name == "func1"
        assert tool_params == {}

    def test_handles_missing_right_matching_bracket(self):
        content = '<tool>[func1(a="1", b="2")]</tool>'

        res = parse_tool_calls(content)

        assert len(res) == 1
        tool_name, tool_params = res[0]
        assert tool_name == "func1"
        assert tool_params == {"a": "1", "b": "2"}

    def test_handles_missing_left_matching_bracket_and_right_matching_bracket(self):
        content = '<tool>func1(a="1", b="2")</tool>'

        res = parse_tool_calls(content)

        assert len(res) == 1
        tool_name, tool_params = res[0]
        assert tool_name == "func1"
        assert tool_params == {"a": "1", "b": "2"}

    def test_handles_multiple_tool_calls(self):
        content = '<tool>[func1(a="1", b="2"), func2(c="3", d="4")]</tool>'

        res = parse_tool_calls(content)

        assert len(res) == 2
        assert res[0] == ("func1", {"a": "1", "b": "2"})
        assert res[1] == ("func2", {"c": "3", "d": "4"})

    def test_handles_multiple_tool_tags_and_text(self):
        content = """
        I should use func1 to do something.
        <tool>[func1(a="1", b="2")]</tool>
        I should use func2 to do something else.
        <tool>[func2(c="3", d="4")]</tool>
        """

        res = parse_tool_calls(content)

        assert len(res) == 2
        assert res[0] == ("func1", {"a": "1", "b": "2"})
        assert res[1] == ("func2", {"c": "3", "d": "4"})


from app19 import main


class TestApp:
    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch):
        """Set up test environment before each test method"""
        # Ensure environment variables are set for tests
        monkeypatch.setenv("GITHUB_API_KEY", "test_key")
        monkeypatch.setenv("LLAMA_STACK_URL", "http://localhost:5000")

    def test_no_github_api_key(self, monkeypatch):
        monkeypatch.delenv("GITHUB_API_KEY", raising=False)

        with pytest.raises(
            ValueError, match="GITHUB_API_KEY is not set in the environment variables"
        ):
            main(
                issue_url="https://github.com/aidando73/bitbucket-syntax-highlighting/issues/67"
            )

    def test_no_llama_stack_url(self, monkeypatch):
        monkeypatch.delenv("LLAMA_STACK_URL", raising=False)

        with pytest.raises(
            ValueError, match="LLAMA_STACK_URL is not set in the environment variables"
        ):
            main(
                issue_url="https://github.com/aidando73/bitbucket-syntax-highlighting/issues/67"
            )


from file_tree_5 import list_files_in_repo
import tempfile
import os
import shutil


class TestFileTree:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment before each test method"""
        # Create a temporary directory for tests
        # Create a new temp directory for each test
        self.test_dir = tempfile.mkdtemp()

        yield

        # Clean up the temp directory after each test
        shutil.rmtree(self.test_dir)

    def test_file_not_found(self):
        with pytest.raises(
            FileNotFoundError, match="File /workspace/does_not_exist does not exist"
        ):
            list_files_in_repo("/workspace/does_not_exist")

    def test_handles_if_file_is_not_in_git(self):
        open(os.path.join(self.test_dir, "file1.txt"), "w").close()

        with pytest.raises(AssertionError, match="not a git repository"):
            list_files_in_repo(self.test_dir)

    def test_default_depth_1(self):
        os.makedirs(os.path.join(self.test_dir, "dir1"))
        os.makedirs(os.path.join(self.test_dir, "dir2"))
        open(os.path.join(self.test_dir, "file1.txt"), "w").close()
        open(os.path.join(self.test_dir, "dir1", "file2.txt"), "w").close()
        add_to_git(self.test_dir)

        res = list_files_in_repo(self.test_dir)

        assert res == ["dir1/", "file1.txt"]

    def test_depth_2(self):
        os.makedirs(os.path.join(self.test_dir, "dir1"))
        os.makedirs(os.path.join(self.test_dir, "dir2"))
        open(os.path.join(self.test_dir, "file1.txt"), "w").close()
        open(os.path.join(self.test_dir, "dir1", "file2.txt"), "w").close()
        open(os.path.join(self.test_dir, "dir2", "file3.txt"), "w").close()
        add_to_git(self.test_dir)

        res = list_files_in_repo(self.test_dir, depth=2)

        assert res == [
            "dir1/",
            "dir1/file2.txt",
            "dir2/",
            "dir2/file3.txt",
            "file1.txt",
        ]


from agent17_5 import translate_path, SANDBOX_DIR


class TestTranslatePath:

    def test_workspace_path(self):
        assert translate_path("/workspace/repo") == os.path.join(SANDBOX_DIR, "repo")

    def test_relative_path(self):
        assert translate_path("repo") == os.path.join(SANDBOX_DIR, "repo")


from agent17_5_1 import execute_tool_call


class TestExecuteToolCall:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment before each test method"""
        self.test_dir = os.path.join(SANDBOX_DIR, "test_repo")
        os.makedirs(self.test_dir)
        with open(os.path.join(self.test_dir, "file.txt"), "w") as f:
            f.write("old content\n\nHello World")

        yield

        shutil.rmtree(self.test_dir)

    def test_list_files_no_path(self):
        res = execute_tool_call("list_files", {})

        assert res == ("error", "ERROR - path not found in tool params: ()")

    def test_list_files_success(self):
        os.makedirs(os.path.join(self.test_dir, "dir1"))
        os.makedirs(os.path.join(self.test_dir, "dir2"))
        open(os.path.join(self.test_dir, "dir1", "file.txt"), "w").close()
        open(os.path.join(self.test_dir, "dir2", "file.txt"), "w").close()
        add_to_git(self.test_dir)

        res = execute_tool_call("list_files", {"path": "/workspace/test_repo"})

        assert res == ("success", "dir1/\ndir2/\nfile.txt")

    def test_list_files_path_not_exists(self):
        res = execute_tool_call(
            "list_files", {"path": "/workspace/test_repo/does_not_exist"}
        )

        assert res == (
            "error",
            f"ERROR - Directory not found: File {self.test_dir}/does_not_exist does not exist",
        )

    def test_list_files_relative_path(self):
        os.makedirs(os.path.join(self.test_dir, "dir1"))
        open(os.path.join(self.test_dir, "dir1", "file.txt"), "w").close()
        add_to_git(self.test_dir)

        res = execute_tool_call("list_files", {"path": "test_repo/dir1"})

        assert res == ("success", "file.txt")

    def test_edit_file_success(self):
        res = execute_tool_call(
            "edit_file",
            {"path": "/workspace/test_repo/file.txt", "new_str": "new content"},
        )

        assert res == ("success", "File successfully updated")
        self.assert_file_content("file.txt", "new content")

    def test_edit_file_error(self):
        res = execute_tool_call(
            "edit_file", {"path": "repo/file.txt", "new_str": "new content"}
        )

        error, error_message = res
        assert error == "error"
        assert (
            "ERROR - File repo/file.txt not found. Please ensure the path is an absolute path and that the file exists."
            == error_message
        )

    def test_edit_file_no_path(self):
        res = execute_tool_call("edit_file", {"new_str": "new content"})

        assert res == (
            "error",
            'ERROR - path not found in tool params: (new_str="new content")',
        )

    def test_edit_file_no_new_str(self):
        res = execute_tool_call("edit_file", {"path": "/workspace/test_repo/file.txt"})

        assert res == (
            "error",
            'ERROR - new_str not found in tool params: (path="/workspace/test_repo/file.txt")',
        )

    def test_edit_file_str_replace(self):
        res = execute_tool_call(
            "edit_file",
            {
                "path": "/workspace/test_repo/file.txt",
                "old_str": "\nHello World",
                "new_str": "Goodbye",
            },
        )

        assert res == ("success", "File successfully updated")
        self.assert_file_content("file.txt", "old content\nGoodbye")

    def assert_file_content(self, path: str, expected_content: str) -> None:
        with open(os.path.join(self.test_dir, path), "r") as f:
            assert f.read() == expected_content


def add_to_git(dir: str) -> None:
    run(
        f"cd {dir} && git init && git add . && git commit -m 'Initial commit'",
        shell=True,
        check=True,
        capture_output=True,
    )
