# Whisper Forced Alignment

An alignment decoder for ![Whisper](https://github.com/openai/whisper). 

Forced alignment operates by analyzing the given audio file alongside a provided text string. Through this process, the model evaluates the likelihood of the speech within the audio accurately representing the specified text.

## Setup

Python and Pytorch requirements are the same as Whisper. The setup and download can be done as:

```bash
pip install git+https://github.com/Warpawn/Whisper-Forced-Alignment
```


## Python usage

To perform forced alignment, utilize the `decode` function within the provided framework. Ensure that `$MODEL_TYPE_WHISP` is specified correctly, such as selecting one of the Whisper models, for example, `large`. The `$audio_file` variable should contain the file path pointing to the audio resource. To activate the forced alignment functionality, set the `alignment_text` parameter accordingly.


```python
import whisper

model = whisper.load_model($MODEL_TYPE_WHISP)

audio = whisper.load_audio($audio_file)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
options = whisper.DecodingOptions(language="en",alignment_text=$current_line)
result = whisper.decode(model, mel, options)

print(result.avg_logprob)
```
