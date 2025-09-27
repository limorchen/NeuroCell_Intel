# filename: Neurocell_Debug_Syntax.py

def explain_syntax_error():
    code_snippet = '''
html = ""
html += f"<li><a href
'''
    try:
        # Try to compile the faulty snippet
        compile(code_snippet, "<string>", "exec")
    except SyntaxError as e:
        print("=== Python raised a SyntaxError ===")
        print(f"Message : {e.msg}")
        print(f"File    : {e.filename}")
        print(f"Line    : {e.lineno}")
        print(f"Offset  : {e.offset}")
        print(f"Text    : {e.text.strip() if e.text else ''}")

        print("\n=== Explanation ===")
        print("The error 'unterminated f-string literal' (or 'EOL while scanning string literal')")
        print("means that Python found an opening quote for a string but never saw the closing one.")
        print("Here, the f-string starts with:")
        print('    html += f"<li><a href')
        print("But there’s no closing quote, so Python thinks the string continues forever.")
        
        print("\n=== How to Fix ===")
        print("Close the string properly and provide valid HTML, e.g.:")
        print('    html += f"<li><a href=\\"https://example.com\\">Example</a></li>"')

if __name__ == "__main__":
    explain_syntax_error()

