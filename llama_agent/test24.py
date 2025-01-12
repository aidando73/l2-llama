from llama_agent.agent34 import replace_content
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
        old_content = dedent("""\
            def foo(self, var):
                if var:
        """)
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

    def test_replace_content2(self):
        old_file_content = dedent("""\
        class Foo:
            @classmethod
            def _scan_iterable_shape(cls, iterable):

                def f(pointer):
                    if not isinstance(pointer, Iterable):
                        return [pointer], ()
                return f(iterable)
        """)
        old_content = """\
            @classmethod
            def _scan_iterable_shape(cls, iterable):

                def f(pointer):
                    if not isinstance(pointer, Iterable):
        """
        new_content = """\
        @classmethod
        def _scan_iterable_shape(cls, iterable):
            def func(pointer):
                if not isinstance(pointer, Iterable):
        """
        new_file_content = replace_content(old_file_content, old_content, new_content)
        assert new_file_content == dedent("""\
        class Foo:
            @classmethod
            def _scan_iterable_shape(cls, iterable):
                def func(pointer):
                    if not isinstance(pointer, Iterable):
                        return [pointer], ()
                return f(iterable)
        """)
