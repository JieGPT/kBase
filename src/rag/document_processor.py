from pathlib import Path
from typing import List, Optional
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Document:
    def __init__(
        self, id: str, content: str, source: str, page: Optional[int] = None, metadata: dict = None
    ):
        self.id = id
        self.content = content
        self.source = source
        self.page = page
        self.metadata = metadata or {}


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def parse_pdf(self, file_path: Path) -> List[tuple]:
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                pages.append((i + 1, text))
        return pages

    def parse_txt(self, file_path: Path) -> List[tuple]:
        with open(file_path, "r", encoding="utf-8") as f:
            return [(None, f.read())]

    def process_document(self, file_path: Path) -> List[Document]:
        suffix = file_path.suffix.lower()
        filename = file_path.name

        if suffix == ".pdf":
            pages = self.parse_pdf(file_path)
        elif suffix == ".txt":
            pages = self.parse_txt(file_path)
        else:
            return []

        documents = []
        for page_num, text in pages:
            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    id=f"{filename}_{page_num or 0}_{i}",
                    content=chunk,
                    source=filename,
                    page=page_num,
                    metadata={"source": filename, "page": page_num, "chunk_index": i},
                )
                documents.append(doc)

        return documents

    def process_directory(
        self, dir_path: str, extensions: List[str] = [".pdf", ".txt"]
    ) -> List[Document]:
        dir_path = Path(dir_path)
        all_documents = []

        for ext in extensions:
            for file_path in dir_path.glob(f"*{ext}"):
                documents = self.process_document(file_path)
                all_documents.extend(documents)

        return all_documents
