from PyPDF2 import PdfReader

def read_file(file_obj):
    """
    Read a Pdf File from a file-like object
    """
    try:
        pdf_reader = PdfReader(file_obj)
        content = ''
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                content += text
        return content
    except Exception as e:
        print(f"Error in Reading PDF: {e}")
        return None


class PdfIngestion:
    def __init__(self, uploaded_files):
        self.uploaded_files = uploaded_files
        self.pdf_content = {}

    def process_file(self, file_obj):
        """
        Process a Single Pdf and store content
        """
        content = read_file(file_obj)
        if content:
            self.pdf_content[file_obj.name] = content
        else:
            print(f"Failed to extract content from {file_obj.name}")

    def load_pdf(self):
        """
        Load Pdf Files sequentially
        """
        for file_obj in self.uploaded_files:
            self.process_file(file_obj)
        return self.pdf_content
