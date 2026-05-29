runLoadQdr:
	python -m storage.load_knowledge_base_to_Qdrant
runLoadKB:
	python -m storage.load_query_index_to_Qdrant	
runChunkSearch:
	python -m chunk_search.main

runChunk:
	python -m top_chunks.main
runSearchChunksTest:
	python -m llm.main

run application:
	python -m app.main