import magic

file_path = "E:\\Major Project\\langchain\\2_news_research_tool_project\\notebooks\\text_loaders_splitters.ipynb"  # Update this path
try:
    print(magic.from_file(file_path, mime=True))
except Exception as e:
    print(f"Error: {e}")
