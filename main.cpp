#include <windows.h>
#include <math.h>
#include <chrono>
#include <iostream>

// Import LAPACK dgemm for verification
extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

// DGEMM Matrix multiplication
void BlasMatrixMultiply(double *matrixA, int aWidth, int aHeight, double *matrixB, int bWidth, int bHeight, double *result)
{
	if (aWidth != bHeight)
	{
		std::cerr << "Multiply matrices failed, bad dimensions." << std::endl;
		return;
	}

	// no transpose
	char op = 'n';
	double alphaScalar = 1.0;
	double betaScalar = 0.0;

	bool success = dgemm_(&op, &op, &aHeight, &bWidth, &aWidth, &alphaScalar, matrixA, &aWidth, matrixB, &aWidth, &betaScalar, result, &aHeight);
	if (!success)
	{
		std::cerr << "dgemm_ failed." << std::endl;
	}
}

//Matrix tile multiplication arguments structure
typedef struct MatrixTileArgs
{
	double *matrixA;
	double *matrixB;
	double *result;
	int size;
	int tileSize;
	int startA;
	int startB;
};

//Computes the residual matrix between 2 matrices of the same size
double* MatrixResidual(double *a, double *b, int width, int height)
{
	double *residual = new double[width * height];
	for (int i = 0; i < width * height; i++)
	{
		residual[i] = a[i] - b[i];
	}

	return residual;
}

//Computes the residual error between 2 matrices of the same size
double MatrixResidualError(double *a, double *b, int width, int height)
{
	double residual = 0;
	for (int i = 0; i < width * height; i++)
	{
		residual += a[i] - b[i];
	}

	return residual;
}

//Generates an array of size N with random elements [0, 2]
double* GenerateRandomMatrix(int size)
{
	double *arr = new double[size];
	for (int i = 0; i < size; i++)
	{
		*(arr + i) = rand() % 12;
	}

	return arr;
}

//Prints a matrix out to the standard output stream
void PrintMatrix(double *matrix, int width, int height)
{
	for (int i = 0; i < width * height; i++)
	{
		if (i % width == 0)
		{
			std::cout << "[ ";
		}

		if (matrix == nullptr)
		{
			std::cout << "NULL ";
		}
		else
		{
			std::cout << matrix[i] << " ";
		}

		if (i % width == width - 1)
		{
			std::cout << "]" << std::endl;
		}
	}
}

//Multiplies the matrix A by the matrix B given the tiled range
double* MatrixMultiply(double *a, int aWidth, int aHeight, double *b, int bWidth, int bHeight, int aStart, int aEnd, int bStart, int bEnd)
{
	// Calculate tile width, height
	int tileHeight = aEnd - aStart;
	int tileWidth = bEnd - bStart;

	// Create array of 0's for tile result
	double *result = new double[tileHeight * tileWidth]{ 0 };

	// Check matrix dimensions
	if (bHeight != aWidth)
	{
		std::cerr << "Bad matrices" << std::endl;
		return nullptr;
	}

	// Loop through columns of B
	// NOTE: To be able to tile we add a start and end index to the range of columns we operate on
	for (int bCol = bStart; bCol < bEnd && bCol < bWidth; bCol++)
	{
		// Loop through rows of A
		// NOTE: To be able to tile we add a start and end index to the range of rows we operate on
		for (int aRow = aStart; aRow < aEnd && aRow < aHeight; aRow++)
		{
			// Multiply current element of current column for each row
			for (int elementIndex = 0; elementIndex < bHeight; elementIndex++)
			{
				// Get element of b column
				double bElement = b[(elementIndex * bWidth) + bCol];

				// Get element of a row
				double aElement = a[(aRow * aWidth) + elementIndex];

				// Calculate index at which we are storing the dot product
				int index = ((aRow - aStart) * tileWidth) + (bCol - bStart);

				// Store dot product in result matrix
				result[index] = result[index] + (aElement * bElement);
			}
		}
	}

	return result;
}

// Threaded matrix multiplication function
DWORD WINAPI MatrixMultiplyByArgs(LPVOID lpParam)
{
	// Cast matrix tile arguments from thread call
	MatrixTileArgs *args = ((MatrixTileArgs*)lpParam);

	// Call the appropriate matrix multiplication and store result
	args->result = MatrixMultiply(args->matrixA, args->size, args->size, args->matrixB, args->size, args->size, args->startA, args->startA + args->tileSize, args->startB, args->startB + args->tileSize);

	// Exit with success
	return 0;
}

// Multiplies 2 squares matrices A and B of the same size using a multithreaded tiled algorithm
double* MatrixMultiplyThreaded(double* a, double *b, int size, int threadCount)
{
	// Check to make sure thread count is between 1 and size of matrix
	if (threadCount <= 0 || threadCount > size * size)
	{
		std::cerr << "Thread count [" << threadCount << "] must be between 1 and the size of the matrix" << std::endl;
		return nullptr;
	}

	// Check to make sure thread count is power of 4
	if ((threadCount & (threadCount - 1)) != 0 || (threadCount & 0xAAAAAAAA))
	{
		// Thread count is not power of 4
		std::cerr << "Thread count [" << threadCount << "] must be power of 4->" << std::endl;
		return nullptr;
	}

	// Calculate tile size and tiles per dimension
	int tileSplits = log(threadCount) / log(4);
	int tileSize = size / pow(2, tileSplits);
	int tilesPerDim = size / tileSize;

	// Create array of pthread_t
	HANDLE *threads = new HANDLE[threadCount];

	// Create array of matrix tile argument pointers
	MatrixTileArgs **argArr = new MatrixTileArgs*[threadCount];

	// Create and run threads
	for (int t = 0; t < threadCount; t++)
	{
		// Calculate the X and Y tile of the matrix to multiply for each thread
		int y = (t / tilesPerDim) * tileSize;
		int x = (t * tileSize) % size;

		// Create matrix tile multiplication parameters
		// NOTE: Arguments must be allocated on the heap, otherwise the structure can be modified on the stack after the scope change
		// and the thread may have invalid data.
		MatrixTileArgs *args = new MatrixTileArgs();
		args->matrixA = a;
		args->matrixB = b;
		args->size = size;
		args->tileSize = tileSize;
		args->startA = y;
		args->startB = x;

		// Try to create and run the threaded matrix tile multiplication
		DWORD threadId;
		HANDLE thread = CreateThread(NULL, 0, MatrixMultiplyByArgs, args, 0, &threadId);
		if (thread == NULL)
		{
			std::cerr << "Failed to start thread " << t << std::endl;

			// Close and delete previous thread resources 
			delete args;
			for (int z = t - 1; z >= 0; z--)
			{
				CloseHandle(threads[z]);
				delete argArr[z];
			}
			delete[] argArr;
			delete[] threads;

			// Return with no result
			return nullptr;
		}

		// Store thread so we can join later
		threads[t] = thread;

		// Store arguments for deletion
		argArr[t] = args;
	}

	// Wait for all threads to finish
	WaitForMultipleObjects(threadCount, threads, TRUE, INFINITE);

	// Create array to store answer
	double *answer = new double[size * size];

	// Let all threads to join, append results
	for (int i = 0; i < threadCount; i++)
	{
		// Calculate the X and Y tile that this thread multiplied
		int y = (i / tilesPerDim) * tileSize;
		int x = (i * tileSize) % size;

		// Wait for the thread to join and retrieve the result
		CloseHandle(threads[i]);

		// Cast result
		double *tile = argArr[i]->result;

		// Copy over tile to answer array
		for (int j = 0; j < tileSize; j++)
		{
			for (int k = 0; k < tileSize; k++)
			{
				int answerIndex = ((j + y) * size) + k + x;
				int tileIndex = (j * tileSize) + k;
				answer[answerIndex] = tile[tileIndex];
			}
		}

		// Delete tile result
		delete[] tile;

		// Delete the arguments for this thread
		delete argArr[i];
	}

	// Delete the thread array
	delete[] threads;

	// Delete the arguments array
	delete[] argArr;

	// Return answer
	return answer;
}

// 
void TestPerformance(int N)
{
	std::cout << "<<< TEST FOR SIZE " << N << " >>>" << std::endl;

	double *matrixA = GenerateRandomMatrix(N * N);
	double *matrixB = GenerateRandomMatrix(N * N);

	double *blasResult = new double[N * N];

	auto start1 = std::chrono::high_resolution_clock::now();
	BlasMatrixMultiply(matrixB, N, N, matrixA, N, N, blasResult);
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> dur1 = end1 - start1;

	std::cout << "BLAS Time = " << dur1.count() << " seconds." << std::endl;

	auto start2 = std::chrono::high_resolution_clock::now();
	double *singleThreadResult = MatrixMultiply(matrixA, N, N, matrixB, N, N, 0, N, 0, N);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> dur2 = end2 - start2;

	std::cout << "Matrix Multiplication (1 Thread) Time = " << dur2.count() << " seconds." << std::endl;
	std::cout << "Matrix Multiplication (1 Thread) Residual Error = " << MatrixResidualError(singleThreadResult, blasResult, N, N) << std::endl;

	//std::cout << "A" << std::endl;
	//PrintMatrix(matrixA, N, N);
	//std::cout << "B" << std::endl;
	//PrintMatrix(matrixB, N, N);

	//std::cout << "result" << std::endl;
	//PrintMatrix(singleThreadResult, 4, 4);
	//std::cout << "blas" << std::endl;
	//PrintMatrix(blasResult, 4, 4);

	delete[] singleThreadResult;

	auto start3 = std::chrono::high_resolution_clock::now();
	double *threadedResult = MatrixMultiplyThreaded(matrixA, matrixB, N, 4);
	auto end3 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> dur3 = end3 - start3;

	std::cout << "Matrix Multiplication (4 Threads) Time = " << dur3.count() << " seconds." << std::endl;
	std::cout << "Matrix Multiplication (4 Threads) Residual Error = " << MatrixResidualError(threadedResult, blasResult, N, N) << std::endl;

	delete[] threadedResult;
	delete[] blasResult;
	delete[] matrixA;
	delete[] matrixB;
}

//Main subroutine
int main()
{
	srand(0);

	int sizes[12] = { 64, 256, 512, 1024, 2048, 3200, 4000, 5000, 5500, 6000, 6500, 7000 };
	for (int i = 0; i < 12; i++)
	{
		TestPerformance(sizes[i]);
	}

	std::cout << "Done. Press any key to exit...";
	std::cin.get();

	return 0;
}
