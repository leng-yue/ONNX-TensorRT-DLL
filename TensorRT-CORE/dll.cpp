#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "parserOnnxConfig.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"
#include <direct.h>
#include <io.h>

using namespace nvinfer1;
samplesCommon::Args gArgs;
#define Debug(x) std::cout << "Line" << __LINE__ << " " << x << std::endl

struct TensorRT {
	IExecutionContext* context;
	ICudaEngine* engine;
	IRuntime* runtime;
};

extern "C" __declspec(dllexport) int ONNX2TRT(char* onnxFileName, char* trtFileName, int batchSize) 
{
	if (_access(onnxFileName, 02) != 0)
	{
		Debug("Can't Read ONNX File");
		return -1;
	}

	// create the builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	if (builder == nullptr) 
	{
		Debug("Create Builder Failure");
		return -2;
	}

	// Now We Have BatchSize Here
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(batchSize);

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

	if (!parser->parseFromFile(onnxFileName, static_cast<int>(gLogger.getReportableSeverity())))
	{
		Debug("Parse ONNX Failure");
		return -3;
	}

	builder->setMaxBatchSize(batchSize);
	builder->setMaxWorkspaceSize(1_GiB);
	config->setMaxWorkspaceSize(1_GiB);
	if (gArgs.runInFp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (gArgs.runInInt8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
	}

	samplesCommon::enableDLA(builder, config, gArgs.useDLACore);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	if (!engine)
	{
		Debug("Engine Build Failure");
		return -4;
	}

	// we can destroy the parser
	parser->destroy();

	// serialize the engine, then close everything down
	IHostMemory* trtModelStream = engine->serialize();

	engine->destroy();
	network->destroy();
	builder->destroy();

	if (!trtModelStream)
	{
		Debug("Serialize Fail");
		return -5;
	}

	ofstream ofs(trtFileName, std::ios::out | std::ios::binary);
	ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
	ofs.close();
	trtModelStream->destroy();

	Debug("Save Success");

	return 0;
}

extern "C" __declspec(dllexport) void* LoadNet(char* trtFileName)
{
	if (_access(trtFileName, 02) != 0)
	{
		Debug("Can't Read TRT File");
		return 0;
	}

	std::ifstream t(trtFileName, std::ios::in | std::ios::binary);
	std::stringstream tempStream;
	tempStream << t.rdbuf();
	t.close();
	Debug("TRT File Loaded");

	tempStream.seekg(0, std::ios::end);
	const int modelSize = tempStream.tellg();
	tempStream.seekg(0, std::ios::beg);
	void* modelMem = malloc(modelSize);
	tempStream.read((char*)modelMem, modelSize);

	IRuntime* runtime = createInferRuntime(gLogger);
	if (runtime == nullptr)
	{
		Debug("Build Runtime Failure");
		return 0;
	}

	if (gArgs.useDLACore >= 0)
	{
		runtime->setDLACore(gArgs.useDLACore);
	}

	ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, nullptr);

	if (engine == nullptr)
	{
		Debug("Build Engine Failure");
		return 0;
	}

	IExecutionContext* context = engine->createExecutionContext();
	if (context == nullptr)
	{
		Debug("Build Context Failure");
		return 0;
	}

	TensorRT* trt = new TensorRT();
	trt->context = context;
	trt->engine = engine;
	trt->runtime = runtime;

	return trt;
}

extern "C" __declspec(dllexport) void ReleaseNet(void* trt) 
{
	TensorRT* curr = (TensorRT*)trt;
	curr->context->destroy();
	curr->engine->destroy();
	curr->runtime->destroy();

	delete curr;
	curr = NULL;
	delete curr;
}

extern "C" __declspec(dllexport) void DoInference(void* trt, char* input_name, char* output_name, float* input, float* output, int input_size, int output_size)
{
	TensorRT* curr = (TensorRT*)trt;

	const ICudaEngine& engine = curr->context->getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()

	const int inputIndex = engine.getBindingIndex(input_name);
	const int outputIndex = engine.getBindingIndex(output_name);

	// DebugP(inputIndex); DebugP(outputIndex);
	// create GPU buffers and a stream

	CHECK(cudaMalloc(&buffers[inputIndex], input_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
	// Because we had specified batch size, so we use enqueueV2
	curr->context->enqueueV2(buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

