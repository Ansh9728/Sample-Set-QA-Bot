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
import os
import pandas as pd

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


class FileIngestion:
    def __init__(self, uploaded_files):
        self.uploaded_files = uploaded_files
        self.documents = []

    def load_files(self):
        """
        Load files of various types (PDF, Excel, CSV, and text)
        from the uploaded_files list.
        """
        for file_obj in self.uploaded_files:
            # Get file extension (lowercase)
            file_ext = os.path.splitext(file_obj.name)[1].lower()

            # Create a temporary file with the same extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(file_obj.read())
                temp_file.flush()

            # Process file based on its extension
            if file_ext == '.pdf':
                loader = PyPDFLoader(temp_file.name)
                loaded = loader.load()  # returns a list of LangChain Document objects
                self.documents.append(loaded)
            elif file_ext in ['.csv']:
                df = pd.read_csv(temp_file.name)
                # Optionally convert DataFrame to string or keep as DataFrame
                self.documents.append(df)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(temp_file.name)
                self.documents.append(df)
            elif file_ext in ['.txt']:
                with open(temp_file.name, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.documents.append(text)
            else:
                print(f"Unsupported file type: {file_ext}")

        return self.documents
