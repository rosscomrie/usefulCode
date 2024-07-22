import time

class TextStreamer:
    def __init__(self, text):
        self.text = text
        
    def stream_data(self):
        for word in self.text.split():
            if '\n' in word:
                yield '  ' + word    # Prefixing a double whitespace for streamlit to render a new line
            else:
                yield word + " "
            time.sleep(0.05)