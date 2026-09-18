using UnityEngine;

#if ULIPSYNC_SENTIS
using Unity.Sentis;
#endif

namespace uLipSync
{

public class AiLipSyncTransformer : AiLipSync
{
    [Header("Transformer Runtime")]
    [Tooltip("When Sentis is unavailable, this component falls back to AiLipSync.fixedModelOutput.")]
    public bool useSentisModel = true;

#if ULIPSYNC_SENTIS
    public ModelAsset modelAsset;
    public BackendType backendType = BackendType.CPU;

    Model _runtimeModel;
    IWorker _worker;
#endif

    protected override float[] RunModel(float[] modelInput)
    {
#if ULIPSYNC_SENTIS
        if (useSentisModel && modelAsset)
        {
            EnsureWorker();
            if (_worker != null)
            {
                // TRANSFORMER INPUT: `modelInput` is the inherited AiLipSync PCM vector.
                // Pass it to ONNX as float32 tensor [1, 1024] by default. The model must use
                // the same sample rate, chronological ordering, and normalization as the
                // recorder and Python training scripts.
                using (var inputTensor = new TensorFloat(
                    new TensorShape(1, modelInput.Length), modelInput))
                {
                    _worker.Execute(inputTensor);
                    var outputTensor = _worker.PeekOutput() as TensorFloat;
                    outputTensor.MakeReadable();
                    var outputValues = outputTensor.ToReadOnlyArray();
                    var output = new float[modelOutputLabels.Length];
                    int count = Mathf.Min(output.Length, outputValues.Length);
                    for (int index = 0; index < count; ++index)
                    {
                        output[index] = outputValues[index];
                    }

                    // The exported PyTorch model returns unrestricted logits. AiLipSync expects
                    // non-negative viseme scores, so convert logits to probabilities here.
                    return Softmax(output);
                }
            }
        }
#endif

        // Temporary fallback: use AiLipSync.fixedModelOutput until an ONNX model and Sentis
        // are configured. The fallback keeps the identical output shape and label order.
        return base.RunModel(modelInput);
    }

#if ULIPSYNC_SENTIS
    void EnsureWorker()
    {
        if (_worker != null || !modelAsset) return;

        _runtimeModel = ModelLoader.Load(modelAsset);
        _worker = WorkerFactory.CreateWorker(backendType, _runtimeModel);
    }

    void OnDestroy()
    {
        _worker?.Dispose();
        _worker = null;
    }
#endif

    float[] Softmax(float[] logits)
    {
        if (logits == null || logits.Length == 0) return logits;

        float maxLogit = float.NegativeInfinity;
        foreach (float logit in logits)
        {
            maxLogit = Mathf.Max(maxLogit, logit);
        }

        float sum = 0f;
        var probabilities = new float[logits.Length];
        for (int index = 0; index < logits.Length; ++index)
        {
            probabilities[index] = Mathf.Exp(logits[index] - maxLogit);
            sum += probabilities[index];
        }

        if (sum <= 0f) return probabilities;
        for (int index = 0; index < probabilities.Length; ++index)
        {
            probabilities[index] /= sum;
        }
        return probabilities;
    }
}

}
