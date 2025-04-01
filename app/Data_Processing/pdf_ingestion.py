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
from langchain_community.document_loaders.csv_loader import CSVLoader

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


import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

class FileIngestion:
    def __init__(self, uploaded_files):
        """
        uploaded_files: a list of file-like objects (e.g., from a file uploader)
        """
        self.uploaded_files = uploaded_files
        self.documents = []

    def load_files(self):
        """
        Load files of various types (PDF, CSV, Excel, and text)
        using LangChain's document loaders.
        """
        for file_obj in self.uploaded_files:
            # Determine file extension (in lowercase)
            file_ext = os.path.splitext(file_obj.name)[1].lower()
            
            # Write the uploaded file content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(file_obj.read())
                temp_file.flush()
            
            # Choose the appropriate loader based on the file extension
            if file_ext == '.pdf':
                loader = PyPDFLoader(temp_file.name)
            elif file_ext == '.csv':
                loader = CSVLoader(file_path=temp_file.name)
            elif file_ext in ['.xls', '.xlsx']:
                loader = UnstructuredExcelLoader(file_path=temp_file.name)
            elif file_ext == '.txt':
                loader = TextLoader(file_path=temp_file.name)
            else:
                print(f"Unsupported file type: {file_ext}")
                continue
            
            # Load the document(s) and extend the list
            docs = loader.load()
            self.documents.extend(docs)
        
        return self.documents
