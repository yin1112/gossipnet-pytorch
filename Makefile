

.PHONY: all

all:  imdb/file_formats/AnnoList_pb2.py


%_pb2.py: %.proto
	protoc --python_out=. $<
 

clean:
	rm -rf all imdb/file_formats/AnnoList_pb2.py