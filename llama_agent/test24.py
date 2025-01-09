from llama_agent.agent24 import replace_content
from textwrap import dedent

class TestReplaceContent:
    def test_replace_content(self):
        old_file_content = dedent("""\
        import os
        import sys
        import time
        import random

        class Foo:
            def foo(self, var):
                if var:
                    print('Hello, world!')
                else:
                    print('Goodbye')
                return "Hello, world!"
        """)
        old_content = "def foo(self, var):"
        new_content = dedent("""\
        def bar(self, is_hello):
            if is_hello:
        """)
        new_file_content = replace_content(old_file_content, old_content, new_content)
        assert new_file_content == dedent("""\
        import os
        import sys
        import time
        import random

        class Foo:
            def bar(self, is_hello):
                if is_hello:
                    print('Hello, world!')
                else:
                    print('Goodbye')
                return "Hello, world!"
        """)
