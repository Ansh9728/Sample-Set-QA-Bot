# from PyPDF2 import PdfReader

# def read_file(file_obj):
#     """
#     Read a Pdf File from a file-like object
#     """
#     try:
#         pdf_reader = PdfReader(file_obj)
#         content = ''
#         for page in pdf_reader.pages:
#             text = page.extract_text()
#             if text:
#                 content += text
#         return content
#     except Exception as e:
#         print(f"Error in Reading PDF: {e}")
#         return None


# class PdfIngestion:
#     def __init__(self, uploaded_files):
#         self.uploaded_files = uploaded_files
#         self.pdf_content = {}

#     def process_file(self, file_obj):
#         """
#         Process a Single Pdf and store content
#         """
#         content = read_file(file_obj)
#         if content:
#             self.pdf_content[file_obj.name] = content
#         else:
#             print(f"Failed to extract content from {file_obj.name}")

#     def load_pdf(self):
#         """
#         Load Pdf Files sequentially
#         """
#         for file_obj in self.uploaded_files:
#             self.process_file(file_obj)
#         return self.pdf_content

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.documents import Document
# from PyPDF2 import PdfReader

# def read_file(file_obj):
#     """
#     Read a Pdf File from a file-like object
#     """
#     try:
#         loader = PyPDFLoader(file_obj)
#         docs = loader.load()
#         return docs
#     except Exception as e:
#         print(f"Error in Reading PDF: {e}")
#         return None


# class PdfIngestion:
#     def __init__(self, uploaded_files):
#         self.uploaded_files = uploaded_files
#         self.pdf_content = {}

#     def process_file(self, file_obj):
#         """
#         Process a Single Pdf and store content
#         """
#         content = read_file(file_obj)
#         if content:
#             self.pdf_content[file_obj.name] = content
#         else:
#             print(f"Failed to extract content from {file_obj.name}")

#     def load_pdf(self):
#         """
#         Load Pdf Files sequentially
#         """
#         document = []
#         for file_obj in self.uploaded_files:
#             docs = self.process_file(file_obj)
#             document.append(docs)

#         return  document

        # documents = []
        # for file_obj in self.uploaded_files:
        #     self.process_file(file_obj)
        #     for filename, content in self.pdf_content.items():
        #         document = Document(
        #             page_content=str(content),
        #             metadata={
                        
        #             }
        #         )
        #         documents.append(document)
        # return documents
from langchain_community.document_loaders import PyPDFLoader
import tempfile

class PdfIngestion:
    def __init__(self, uploaded_files):
        self.uploaded_files = uploaded_files
        self.pdf_documents = []

    def load_pdf(self):
        """
        Load the Pdf Document as a Langchain Document loader

        """
        for file_obj in self.uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_obj.read())
                temp_file.flush()  # Ensure all data is written

            # Load the PDF from the temporary file path
            loader = PyPDFLoader(temp_file.name)
            # print(loader.load())+
            self.pdf_documents.append(loader.load())

        return self.pdf_documents