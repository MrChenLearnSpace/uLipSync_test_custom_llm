using System;
using System.Collections.Generic;
using UnityEngine;

namespace uLipSync
{

[RequireComponent(typeof(AudioSource))]
public class AiLipSync : MonoBehaviour
{
    [Header("Audio Input")]
    public uLipSyncAudioSource audioSourceProxy;
    [Min(1)] public int modelInputSampleRate = 16000;
    [Min(1)] public int modelInputSampleCount = 1600;
    [Range(0.01f, 0.2f)] public float inferenceInterval = 0.02f;
    [Range(0f, 0.1f)] public float silenceVolumeThreshold = 0.005f;

    [Header("Temporary Fixed Model Output")]
    public string[] modelOutputLabels = { "A", "I", "U", "E", "O" };
    public float[] fixedModelOutput = { 1f, 0f, 0f, 0f, 0f };

    [Header("Output")]
    public LipSyncUpdateEvent onLipSyncUpdate = new LipSyncUpdateEvent();

    public float[] latestModelInput => _modelInput;
    public LipSyncInfo result { get; private set; }

    AudioSource _audioSource;
    uLipSyncAudioSource _currentAudioSourceProxy;
    readonly object _lockObject = new object();

    float[] _audioRingBuffer;
    float[] _modelInput;
    int _writeIndex;
    bool _hasAudioData;
    float _nextInferenceTime;

    void Awake()
    {
        UpdateAudioSource();
        UpdateAudioSourceProxy();
    }

    void OnEnable()
    {
        EnsureBuffers();
    }

    void OnDisable()
    {
        if (_currentAudioSourceProxy)
        {
            _currentAudioSourceProxy.onAudioFilterRead.RemoveListener(OnDataReceived);
            _currentAudioSourceProxy = null;
        }
    }

    void Update()
    {
        EnsureBuffers();
        UpdateAudioSource();
        UpdateAudioSourceProxy();

        if (!_hasAudioData || Time.unscaledTime < _nextInferenceTime) return;

        _nextInferenceTime = Time.unscaledTime + inferenceInterval;
        CopyAudioToModelInput();

        var rawVolume = GetRmsVolume(_modelInput);
        var modelOutput = RunModel(_modelInput);
        result = CreateLipSyncInfo(modelOutput, rawVolume);
        onLipSyncUpdate?.Invoke(result);
    }

    void EnsureBuffers()
    {
        int inputSampleRate = Mathf.Max(modelInputSampleRate, 1);
        int inputSampleCount = Mathf.Max(modelInputSampleCount, 1);
        int audioSampleRate = Mathf.Max(AudioSettings.outputSampleRate, 1);
        int ringBufferLength = Mathf.CeilToInt((float)inputSampleCount / inputSampleRate * audioSampleRate);

        if (_modelInput != null && _modelInput.Length == inputSampleCount &&
            _audioRingBuffer != null && _audioRingBuffer.Length == ringBufferLength)
        {
            return;
        }

        lock (_lockObject)
        {
            _modelInput = new float[inputSampleCount];
            _audioRingBuffer = new float[ringBufferLength];
            _writeIndex = 0;
            _hasAudioData = false;
        }
    }

    void UpdateAudioSource()
    {
        if (_audioSource) return;
        _audioSource = GetComponent<AudioSource>();
    }

    void UpdateAudioSourceProxy()
    {
        if (audioSourceProxy == _currentAudioSourceProxy) return;

        if (_currentAudioSourceProxy)
        {
            _currentAudioSourceProxy.onAudioFilterRead.RemoveListener(OnDataReceived);
        }

        if (audioSourceProxy)
        {
            audioSourceProxy.onAudioFilterRead.AddListener(OnDataReceived);
        }

        _currentAudioSourceProxy = audioSourceProxy;
    }

    void OnAudioFilterRead(float[] input, int channels)
    {
        if (audioSourceProxy) return;
        OnDataReceived(input, channels);
    }

    public void OnDataReceived(float[] input, int channels)
    {
        if (input == null || channels <= 0 || _audioRingBuffer == null) return;

        lock (_lockObject)
        {
            int length = _audioRingBuffer.Length;
            for (int index = 0; index < input.Length; index += channels)
            {
                _audioRingBuffer[_writeIndex] = input[index];
                _writeIndex = (_writeIndex + 1) % length;
            }
            _hasAudioData = true;
        }
    }

    void CopyAudioToModelInput()
    {
        lock (_lockObject)
        {
            int inputLength = _modelInput.Length;
            int sourceLength = _audioRingBuffer.Length;

            // MODEL INPUT LOCATION: `_modelInput` (also exposed by `latestModelInput`).
            // MODEL INPUT FORMAT: float32 PCM, mono, chronological order (oldest -> newest),
            // normalized to Unity audio's usual [-1, 1] range. Its fixed tensor shape is
            // [1, modelInputSampleCount], currently [1, 1600], at modelInputSampleRate Hz
            // (currently 16 kHz, representing the latest 100 ms of audio).
            for (int targetIndex = 0; targetIndex < inputLength; ++targetIndex)
            {
                float sourcePosition = inputLength == 1 ? 0f :
                    (float)targetIndex * (sourceLength - 1) / (inputLength - 1);
                int lowerIndex = Mathf.FloorToInt(sourcePosition);
                int upperIndex = Mathf.Min(lowerIndex + 1, sourceLength - 1);
                float fraction = sourcePosition - lowerIndex;
                int ringLowerIndex = (_writeIndex + lowerIndex) % sourceLength;
                int ringUpperIndex = (_writeIndex + upperIndex) % sourceLength;
                _modelInput[targetIndex] = Mathf.Lerp(
                    _audioRingBuffer[ringLowerIndex],
                    _audioRingBuffer[ringUpperIndex],
                    fraction);
            }
        }
    }

    float[] RunModel(float[] modelInput)
    {
        // MODEL INFERENCE REPLACEMENT POINT:
        // Replace this method with Sentis, ONNX Runtime, or a native inference call after the
        // trained model is ready. Pass `modelInput` directly as the tensor described above.
        // The temporary implementation intentionally has no inference dependency and returns
        // the Inspector-configured, fixed output vector below.

        // MODEL OUTPUT LOCATION: the return value of this method.
        // MODEL OUTPUT FORMAT: float32 vector with shape [1, modelOutputLabels.Length],
        // currently [1, 5]. Index i corresponds to modelOutputLabels[i] in this exact order:
        // A, I, U, E, O. Values must be non-negative scores or probabilities; they do not need
        // to sum to 1 because CreateLipSyncInfo normalizes them into LipSyncInfo.phonemeRatios.
        return fixedModelOutput;
    }

    LipSyncInfo CreateLipSyncInfo(float[] modelOutput, float rawVolume)
    {
        var phonemeRatios = new Dictionary<string, float>();
        float scoreSum = 0f;
        string mainPhoneme = "";
        float maxScore = 0f;

        if (rawVolume >= silenceVolumeThreshold && modelOutput != null && modelOutputLabels != null)
        {
            int count = Mathf.Min(modelOutput.Length, modelOutputLabels.Length);
            for (int index = 0; index < count; ++index)
            {
                string phoneme = modelOutputLabels[index];
                float score = Mathf.Max(0f, modelOutput[index]);
                if (string.IsNullOrEmpty(phoneme) || score <= 0f) continue;

                if (phonemeRatios.ContainsKey(phoneme))
                {
                    phonemeRatios[phoneme] += score;
                }
                else
                {
                    phonemeRatios.Add(phoneme, score);
                }

                scoreSum += score;
                if (score > maxScore)
                {
                    maxScore = score;
                    mainPhoneme = phoneme;
                }
            }
        }

        if (scoreSum > 0f)
        {
            var phonemes = new List<string>(phonemeRatios.Keys);
            foreach (var phoneme in phonemes)
            {
                phonemeRatios[phoneme] /= scoreSum;
            }
        }

        return new LipSyncInfo()
        {
            phoneme = mainPhoneme,
            rawVolume = rawVolume,
            volume = NormalizeVolume(rawVolume),
            phonemeRatios = phonemeRatios,
        };
    }

    float NormalizeVolume(float rawVolume)
    {
        if (rawVolume <= 0f) return 0f;

        float logVolume = Mathf.Log10(rawVolume);
        float normalizedVolume = (logVolume - Common.DefaultMinVolume) /
            (Common.DefaultMaxVolume - Common.DefaultMinVolume);
        return Mathf.Clamp01(normalizedVolume);
    }

    float GetRmsVolume(float[] samples)
    {
        if (samples == null || samples.Length == 0) return 0f;

        float sum = 0f;
        foreach (float sample in samples)
        {
            sum += sample * sample;
        }
        return Mathf.Sqrt(sum / samples.Length);
    }
}

}
