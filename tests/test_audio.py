import sys
import types
import wave

import numpy as np
import pytest

from animengine.audio import AudioMixer, Sound, playback, synthesize_thump
from animengine.audio.export import WaveWriter
from animengine.audio.playback import DEFAULT_AUDIO_DELAY, AudioPlayer


def test_thump_is_normalized_and_has_smooth_ends():
    sound = synthesize_thump()
    assert sound.sample_rate == 48_000
    assert len(sound.samples) == 8640
    assert sound.samples[0] == 0
    assert sound.samples[-1] == 0
    assert np.max(np.abs(sound.samples)) == pytest.approx(1)


def test_mix_is_sample_accurate_across_blocks_and_unchanged_below_knee():
    sound = Sound([0.0, 0.25, 0.5, 0.25, 0.0], 10)
    mixer = AudioMixer(10)
    mixer.schedule(sound, 0.2, gain=0.5)
    actual = np.concatenate((mixer.render(3), mixer.render(4), mixer.render(3)))
    assert np.allclose(actual, [0, 0, 0, 0.125, 0.25, 0.125, 0, 0, 0, 0])
    assert mixer.sample_position == 10


def test_many_overlapping_sounds_are_bounded_and_do_not_hard_clip():
    mixer = AudioMixer(100, max_voices=50)
    sound = Sound(np.ones(20), 100)
    for _ in range(50):
        mixer.schedule(sound, 0, gain=1)
    samples = mixer.render(20)
    assert np.all(samples > 0.6)
    assert np.max(samples) <= np.float32(0.85)
    assert np.max(samples) < 1
    assert mixer.dropped_voices == 0


def test_limiter_recovers_smoothly_after_a_loud_peak():
    mixer = AudioMixer(1000)
    mixer.schedule(Sound([1, 0.1, 0.1, 0.1], 1000), 0)
    samples = mixer.render(4)
    assert 0.6 < samples[0] <= 0.85
    assert 0 < samples[1] < samples[2] < samples[3] < 0.1


def test_voice_limit_counts_overflow_and_releases_finished_voices():
    mixer = AudioMixer(10, max_voices=1)
    sound = Sound([0.5, 0.5], 10)
    mixer.schedule(sound, 0)
    mixer.schedule(sound, 0.1)
    mixer.schedule(sound, 0.2)
    assert np.allclose(mixer.render(4), [0.5, 0.5, 0.5, 0.5])
    assert mixer.dropped_voices == 1


def test_late_event_is_reported_and_plays_at_next_sample():
    mixer = AudioMixer(10)
    sound = Sound([0.5], 10)
    mixer.render(5)
    with pytest.warns(RuntimeWarning, match="late"):
        mixer.schedule(sound, 0.2)
    assert mixer.late_events == 1
    assert np.allclose(mixer.render(2), [0.5, 0])


def test_block_size_does_not_change_mix():
    sound = synthesize_thump(1000, frequency=60, duration=0.2)
    mixers = [AudioMixer(1000), AudioMixer(1000)]
    for mixer in mixers:
        for time in (0, 0.003, 0.025, 0.19):
            mixer.schedule(sound, time, 0.3)
    whole = mixers[0].render(500)
    blocks = np.concatenate([mixers[1].render(n) for n in (1, 25, 37, 7, 430)])
    assert np.array_equal(whole, blocks)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_schedule_rejects_bad_time_and_gain(value):
    mixer = AudioMixer(10)
    sound = Sound([0], 10)
    with pytest.raises(ValueError):
        mixer.schedule(sound, value)
    with pytest.raises(ValueError):
        mixer.schedule(sound, 0, value)


def test_sample_rates_must_match():
    with pytest.raises(ValueError, match="sample rates"):
        AudioMixer(20).schedule(Sound([0], 10), 0)


def test_wave_export_fades_the_end_without_changing_duration(tmp_path):
    mixer = AudioMixer(100)
    mixer.schedule(Sound(np.full(20, 0.5), 100), 0)
    path = tmp_path / "mix.wav"
    with WaveWriter(path, mixer, 20, fade_seconds=0.05) as writer:
        writer.write_until(10)
        writer.write_until(20)
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 100
        assert wav.getnframes() == 20
        samples = np.frombuffer(wav.readframes(20), dtype="<i2") / 32767
    assert np.allclose(samples[:15], 0.5, atol=1e-4)
    assert np.all(np.diff(samples[-6:]) <= 0)
    assert samples[-1] == 0


def test_preview_callback_delays_audio_and_reports_status():
    mixer = AudioMixer(10)
    mixer.schedule(Sound([0.5, 0.5, 0.5], 10), 0)
    player = AudioPlayer(mixer, delay=0.2, blocksize=4)
    output = np.empty((4, 1), dtype=np.float32)
    player._callback(output, 4, None, "underflow")
    assert np.allclose(output[:, 0], [0, 0, 0.5, 0.5])
    with pytest.warns(RuntimeWarning, match="underflow"):
        player.check()
    player._callback(output, 4, None, None)
    assert np.allclose(output[:, 0], [0.5, 0, 0, 0])


def test_default_preview_scheduling_buffer_is_60_ms():
    mixer = AudioMixer(1000)
    mixer.schedule(Sound([0.5], 1000), 0)
    player = AudioPlayer(mixer)
    output = np.empty((70, 1), dtype=np.float32)
    player._callback(output, 70, None, None)
    assert DEFAULT_AUDIO_DELAY == 0.06
    assert np.all(output[:60] == 0)
    assert output[60, 0] == pytest.approx(0.5)


def test_preview_reports_dropped_voices():
    mixer = AudioMixer(10, max_voices=1)
    sound = Sound([0.5, 0.5], 10)
    mixer.schedule(sound, 0)
    mixer.schedule(sound, 0)
    player = AudioPlayer(mixer, delay=0)
    output = np.empty((2, 1), dtype=np.float32)
    player._callback(output, 2, None, None)
    with pytest.warns(RuntimeWarning, match="dropped 1"):
        player.check()


def test_preview_opens_and_closes_audio_stream(monkeypatch):
    streams = []

    class FakeStream:
        def __init__(self, **options):
            self.options = options
            self.started = False
            self.stopped = False
            self.closed = False
            streams.append(self)

        def start(self):
            self.started = True
            self.options["callback"](np.empty((4, 1), dtype=np.float32), 4, None, None)

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules, "sounddevice",
        types.SimpleNamespace(
            OutputStream=FakeStream, query_hostapis=lambda: [], PortAudioError=RuntimeError,
        ),
    )
    with AudioPlayer(AudioMixer(10), delay=0, blocksize=4) as player:
        player.check()
        assert streams[0].started
    assert streams[0].stopped
    assert streams[0].closed


def test_callback_failure_is_reported_to_preview(monkeypatch):
    class CallbackAbort(Exception):
        pass

    monkeypatch.setitem(
        sys.modules, "sounddevice", types.SimpleNamespace(CallbackAbort=CallbackAbort)
    )
    mixer = AudioMixer(10)

    def fail(_):
        raise ValueError("mixing failed")

    monkeypatch.setattr(mixer, "render", fail)
    player = AudioPlayer(mixer, delay=0)
    output = np.empty((2, 1), dtype=np.float32)
    with pytest.raises(CallbackAbort):
        player._callback(output, 2, None, None)
    assert np.all(output == 0)
    with pytest.raises(RuntimeError, match="Audio playback failed") as error:
        player.check()
    assert isinstance(error.value.__cause__, ValueError)


def test_preview_checks_for_callback_failure_during_shutdown(monkeypatch):
    class CallbackAbort(Exception):
        pass

    class FakeStream:
        def __init__(self, **options):
            self.callback = options["callback"]
            self.closed = False

        def start(self):
            pass

        def stop(self):
            with pytest.raises(CallbackAbort):
                self.callback(np.empty((1, 1), dtype=np.float32), 1, None, None)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules, "sounddevice",
        types.SimpleNamespace(
            OutputStream=FakeStream, CallbackAbort=CallbackAbort,
            query_hostapis=lambda: [], PortAudioError=RuntimeError,
        ),
    )
    mixer = AudioMixer(10)

    def fail(_):
        raise ValueError("late callback failure")

    monkeypatch.setattr(mixer, "render", fail)
    player = AudioPlayer(mixer, delay=0)
    with pytest.raises(RuntimeError, match="Audio playback failed"):
        with player:
            pass


def test_windows_prefers_wasapi_only_for_the_default_mme_device(monkeypatch):
    apis = [
        {"name": "MME", "default_output_device": 3},
        {"name": "Windows WASAPI", "default_output_device": 10},
    ]
    device = types.SimpleNamespace(
        query_hostapis=lambda: apis, default=types.SimpleNamespace(device=(1, 3)),
    )
    monkeypatch.setattr(playback.sys, "platform", "win32")
    assert playback._preferred_output_device(device) == 10
    device.default.device = (1, 8)
    assert playback._preferred_output_device(device) is None
    device.default.device = (1, 3)
    apis[1]["default_output_device"] = -1
    assert playback._preferred_output_device(device) is None
    monkeypatch.setattr(playback.sys, "platform", "linux")
    assert playback._preferred_output_device(device) is None


def test_preview_falls_back_when_automatic_wasapi_unavailable(monkeypatch):
    class PortAudioError(Exception):
        pass

    opened = []

    class FakeStream:
        def __init__(self, **options):
            self.device = options["device"]
            self.closed = False
            opened.append(self)

        def start(self):
            if self.device == 10:
                raise PortAudioError("device unavailable")

        def stop(self):
            pass

        def close(self):
            self.closed = True

    api = types.SimpleNamespace(
        OutputStream=FakeStream, PortAudioError=PortAudioError,
        query_hostapis=lambda: [
            {"name": "MME", "default_output_device": 3},
            {"name": "Windows WASAPI", "default_output_device": 10},
        ],
        default=types.SimpleNamespace(device=(1, 3)),
    )
    monkeypatch.setattr(playback.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "sounddevice", api)
    with pytest.warns(RuntimeWarning, match="WASAPI output unavailable"):
        with AudioPlayer(AudioMixer()) as player:
            player.check()
    assert [stream.device for stream in opened] == [10, None]
    assert all(stream.closed for stream in opened)


def test_explicit_audio_device_is_not_overridden(monkeypatch):
    opened = []

    class FakeStream:
        def __init__(self, **options):
            opened.append(options["device"])

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    api = types.SimpleNamespace(OutputStream=FakeStream, PortAudioError=RuntimeError)
    monkeypatch.setitem(sys.modules, "sounddevice", api)
    with AudioPlayer(AudioMixer(), device=8):
        pass
    assert opened == [8]
