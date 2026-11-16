all: bin/server

bin/server: bin/Main.o bin/Model.o bin/MemoryBlock.o bin/GPU.o bin/Request.o bin/SchedulingPolicy.o bin/Server.o
	g++ -o bin/server bin/Main.o bin/Model.o bin/MemoryBlock.o bin/GPU.o bin/Request.o bin/SchedulingPolicy.o bin/Server.o

bin/Main.o: src/Main.cpp include/*.h
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o

bin/Model.o: src/Model.cpp include/*.h
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/Model.cpp -o bin/Model.o

bin/MemoryBlock.o: src/MemoryBlock.cpp include/*.h
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/MemoryBlock.cpp -o bin/MemoryBlock.o

bin/GPU.o: src/GPU.cpp include/*.h
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/GPU.cpp -o bin/GPU.o

bin/Request.o: src/Request.cpp include/*.h
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/Request.cpp -o bin/Request.o

bin/SchedulingPolicy.o: src/SchedulingPolicy.cpp include/*.h
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/SchedulingPolicy.cpp -o bin/SchedulingPolicy.o

bin/Server.o: src/Server.cpp include/*.h
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/Server.cpp -o bin/Server.o

clean:x
	rm -f bin/*.o bin/server

.PHONY: all clean