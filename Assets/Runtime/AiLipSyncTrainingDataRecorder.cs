using System;
using System.IO;
using UnityEngine;

namespace uLipSync
{

public class AiLipSyncTrainingDataRecorder : MonoBehaviour
{
    [Header("Legacy Label Source")]
    public uLipSync legacyLipSync;

    [Header("Recorded Model Input")]
    [Min(1)] public int modelInputSampleRate = 16000;
    [Min(1)] public int modelInputSampleCount = 1024;
    [Range(0.01f, 0.2f)] public float recordInterval = 1f / 60f;
    public bool recordOnlyWhenVoice = true;
    [Range(0f, 0.1f)] public float minimumRawVolume = 0.005f;

    [Header("Recorded Model Output")]
    public string[] modelOutputLabels = { "A", "I", "U", "E", "O" };

    [Header("Session")]
    public bool recordOnEnable = false;
    public string outputDirectoryName = "uLipSyncTrainingData";
    [SerializeField] string _lastOutputDirectory;
    [SerializeField] int _recordCount;

    BinaryWriter _inputWriter;
    BinaryWriter _labelWriter;
    StreamWriter _manifestWriter;
    float[] _modelInput;
    float[] _modelOutput;
    bool _isRecording;
    double _nextRecordTime;

    public bool isRecording => _isRecording;
    public string lastOutputDirectory => _lastOutputDirectory;
    public int recordCount => _recordCount;

    void Reset()
    {
        legacyLipSync = GetComponent<uLipSync>();
    }

    void OnEnable()
    {
        if (!legacyLipSync)
        {
            legacyLipSync = GetComponent<uLipSync>();
        }

        if (legacyLipSync)
        {
            legacyLipSync.onLipSyncUpdate.AddListener(OnLegacyLipSyncUpdate);
        }

        if (recordOnEnable)
        {
            StartRecording();
        }
    }

    void OnDisable()
    {
        if (legacyLipSync)
        {
            legacyLipSync.onLipSyncUpdate.RemoveListener(OnLegacyLipSyncUpdate);
        }

        StopRecording();
    }

    [ContextMenu("Start Recording")]
    public void StartRecording()
    {
        if (_isRecording) return;

        if (!legacyLipSync)
        {
            Debug.LogError("AiLipSyncTrainingDataRecorder requires a uLipSync component.", this);
            return;
        }

        ValidateInputConfiguration();

        int inputCount = Mathf.Max(modelInputSampleCount, 1);
        int outputCount = modelOutputLabels == null ? 0 : modelOutputLabels.Length;
        if (outputCount == 0)
        {
            Debug.LogError("Model Output Labels must contain at least one viseme label.", this);
            return;
        }

        _modelInput = new float[inputCount];
        _modelOutput = new float[outputCount];
        _recordCount = 0;
        _nextRecordTime = AudioSettings.dspTime;
        _lastOutputDirectory = Path.Combine(
            Application.persistentDataPath,
            outputDirectoryName,
            DateTime.Now.ToString("yyyyMMdd-HHmmss"));

        Directory.CreateDirectory(_lastOutputDirectory);
        _inputWriter = new BinaryWriter(File.Open(
            Path.Combine(_lastOutputDirectory, "inputs.f32"),
            FileMode.Create,
            FileAccess.Write,
            FileShare.Read));
        _labelWriter = new BinaryWriter(File.Open(
            Path.Combine(_lastOutputDirectory, "labels.f32"),
            FileMode.Create,
            FileAccess.Write,
            FileShare.Read));
        _manifestWriter = new StreamWriter(Path.Combine(_lastOutputDirectory, "manifest.jsonl"));
        _isRecording = true;

        WriteMetadata();
        Debug.Log($"Started lip-sync training-data recording: {_lastOutputDirectory}", this);
    }

    [ContextMenu("Stop Recording")]
    public void StopRecording()
    {
        if (!_isRecording) return;

        _inputWriter?.Dispose();
        _labelWriter?.Dispose();
        _manifestWriter?.Dispose();
        _inputWriter = null;
        _labelWriter = null;
        _manifestWriter = null;
        _isRecording = false;

        WriteMetadata();
        Debug.Log($"Stopped lip-sync training-data recording. Frames: {_recordCount}", this);
    }

    // This method is automatically subscribed to uLipSync.onLipSyncUpdate in OnEnable.
    // It can also be assigned manually in the Inspector if the recorder is on another object.
    public void OnLegacyLipSyncUpdate(LipSyncInfo info)
    {
        if (!_isRecording || AudioSettings.dspTime < _nextRecordTime) return;
        if (recordOnlyWhenVoice && info.rawVolume < minimumRawVolume) return;
        if (!legacyLipSync.TryCopyLastAnalyzedAudio(out var legacyInput, out _)) return;

        _nextRecordTime = AudioSettings.dspTime + recordInterval;
        ResampleToModelInput(legacyInput, _modelInput);
        CreateModelOutput(info, _modelOutput);

        if (recordOnlyWhenVoice && Sum(_modelOutput) <= 0f) return;

        // DATASET INPUT FILE: inputs.f32
        // FORMAT: consecutive little-endian float32 rows, each row has exactly
        // `modelInputSampleCount` values. Values are mono PCM in chronological order
        // (oldest -> newest), resampled to `modelInputSampleRate` Hz and normally in [-1, 1].
        foreach (float sample in _modelInput)
        {
            _inputWriter.Write(sample);
        }

        // DATASET OUTPUT FILE: labels.f32
        // FORMAT: consecutive little-endian float32 rows. Row i has one non-negative,
        // normalized viseme score for modelOutputLabels[i]. The default fixed order is
        // [A, I, U, E, O], which matches the existing uLipSync BlendShape convention.
        foreach (float score in _modelOutput)
        {
            _labelWriter.Write(score);
        }

        _manifestWriter.WriteLine(JsonUtility.ToJson(new DatasetRecord()
        {
            index = _recordCount,
            dspTime = AudioSettings.dspTime,
            mainPhoneme = info.phoneme,
            rawVolume = info.rawVolume,
            normalizedVolume = info.volume,
        }));
        _recordCount++;
    }

    void ResampleToModelInput(float[] source, float[] destination)
    {
        if (source == null || source.Length == 0)
        {
            Array.Clear(destination, 0, destination.Length);
            return;
        }

        int sourceLength = source.Length;
        for (int targetIndex = 0; targetIndex < destination.Length; ++targetIndex)
        {
            float sourcePosition = destination.Length == 1 ? 0f :
                (float)targetIndex * (sourceLength - 1) / (destination.Length - 1);
            int lowerIndex = Mathf.FloorToInt(sourcePosition);
            int upperIndex = Mathf.Min(lowerIndex + 1, sourceLength - 1);
            destination[targetIndex] = Mathf.Lerp(
                source[lowerIndex],
                source[upperIndex],
                sourcePosition - lowerIndex);
        }
    }

    void ValidateInputConfiguration()
    {
        var profile = legacyLipSync.profile;
        if (!profile) return;

        if (modelInputSampleRate != profile.targetSampleRate ||
            modelInputSampleCount != profile.sampleCount)
        {
            Debug.LogWarning(
                "Recorder input shape differs from the uLipSync Profile. This stretches or " +
                "compresses the analysis window. Use matching sample rate and sample count " +
                "unless the mismatch is intentional.",
                this);
        }
    }

    void CreateModelOutput(LipSyncInfo info, float[] output)
    {
        Array.Clear(output, 0, output.Length);

        if (info.phonemeRatios != null)
        {
            for (int index = 0; index < output.Length; ++index)
            {
                info.phonemeRatios.TryGetValue(modelOutputLabels[index], out output[index]);
                output[index] = Mathf.Max(0f, output[index]);
            }
        }

        float sum = Sum(output);
        if (sum <= 0f && !string.IsNullOrEmpty(info.phoneme))
        {
            for (int index = 0; index < output.Length; ++index)
            {
                if (modelOutputLabels[index] == info.phoneme)
                {
                    output[index] = 1f;
                    sum = 1f;
                    break;
                }
            }
        }

        if (sum <= 0f) return;
        for (int index = 0; index < output.Length; ++index)
        {
            output[index] /= sum;
        }
    }

    float Sum(float[] values)
    {
        float sum = 0f;
        foreach (float value in values)
        {
            sum += value;
        }
        return sum;
    }

    void WriteMetadata()
    {
        if (string.IsNullOrEmpty(_lastOutputDirectory) || !Directory.Exists(_lastOutputDirectory)) return;

        var metadata = new DatasetMetadata()
        {
            formatVersion = 1,
            recordCount = _recordCount,
            inputFile = "inputs.f32",
            inputDtype = "float32-le",
            inputShape = new[] { modelInputSampleCount },
            inputSampleRate = modelInputSampleRate,
            outputFile = "labels.f32",
            outputDtype = "float32-le",
            outputShape = new[] { modelOutputLabels == null ? 0 : modelOutputLabels.Length },
            outputLabels = modelOutputLabels,
            manifestFile = "manifest.jsonl",
        };

        File.WriteAllText(
            Path.Combine(_lastOutputDirectory, "metadata.json"),
            JsonUtility.ToJson(metadata, true));
    }

    [Serializable]
    class DatasetMetadata
    {
        public int formatVersion;
        public int recordCount;
        public string inputFile;
        public string inputDtype;
        public int[] inputShape;
        public int inputSampleRate;
        public string outputFile;
        public string outputDtype;
        public int[] outputShape;
        public string[] outputLabels;
        public string manifestFile;
    }

    [Serializable]
    class DatasetRecord
    {
        public int index;
        public double dspTime;
        public string mainPhoneme;
        public float rawVolume;
        public float normalizedVolume;
    }
}

}
