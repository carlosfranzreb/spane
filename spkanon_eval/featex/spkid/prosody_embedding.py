import parselmouth
import numpy as np
import logging
import torch
from omegaconf import OmegaConf
from spkanon_eval.component_definitions import InferComponent

LOGGER = logging.getLogger("progress")


class ProsodyEmbedding(InferComponent):

    def __init__(self, config: OmegaConf, device: str) -> None:
        """
        Initialize the prosody embedding model with the given configuration from ecapa.yaml
        """
        self.device = device
        self.config = config
        self.pitch_floor = config.pitch_floor
        self.pitch_floor_for_segmentation = config.pitch_floor_for_segmentation
        self.pitch_ceiling = config.pitch_ceiling
        self.time_step = config.time_step
        self.sampling_frequency = config.sampling_frequency
        self.feature_segments = config.get("feature_segments")
        self.segmentation_method = config.get("segmentation_method")

    # set the device to run the model on
    def to(self, device: str) -> None:
        self.device = device

    @torch.inference_mode()
    def run(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """
        Return speaker embeddings for the given batch of utterances.

        Args:
            batch: A list of three tensors in the following order:
            1. waveforms with shape (batch_size, n_samples)
            2. waveform lengths with shape (batch_size), as integers
            3. waveform speaker IDs with shape (batch_size), as integers

        Returns:
            A tensor containing the prosody embeddings with shape
            (batch_size, embedding_dim).
        """

        waveforms = batch[0].cpu().numpy()
        embeddings = []

        for waveform in waveforms:
            sound = parselmouth.Sound(
                values=waveform, sampling_frequency=self.sampling_frequency
            )
            embedding = self.create_embedding(
                sound, segmentation_method=self.segmentation_method
            )
            embeddings.append(embedding)

        embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32).to(
            batch[0].device
        )
        # print(f"output shape: {embeddings_tensor.shape}")
        # print(f"output: {embeddings_tensor}")
        return embeddings_tensor

    # approximation of the voiced segments based on f0
    def pitch_and_intensity_contour_segmentation_based_on_pitch(self, sound):

        # create pitch contour for the whole audio
        global_pitch_contour = sound.to_pitch_ac(
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
            time_step=self.time_step,
        )

        global_pitch_values = global_pitch_contour.selected_array["frequency"]
        global_pitch_contour_times = global_pitch_contour.xs()

        # create intensity contour for the whole audio
        intensity = sound.to_intensity(
            minimum_pitch=self.pitch_floor, subtract_mean=True, time_step=self.time_step
        )

        global_intensity_values = intensity.values[0]
        global_intensity_times = intensity.xs()

        # detecting voiced segments based on the paper to approximate words
        pitch_to_segment = sound.to_pitch_ac(
            time_step=0.01,
            pitch_floor=self.pitch_floor_for_segmentation,
            pitch_ceiling=500.0,
            max_number_of_candidates=5,
            silence_threshold=0.03,
            voicing_threshold=0.6,
            octave_cost=0.01,
            octave_jump_cost=0.6,
            voiced_unvoiced_cost=0.14,
        )
        seg_pitch_values = pitch_to_segment.selected_array["frequency"]
        seg_pitch_times = pitch_to_segment.xs()

        # voiced segemts wehre f0 > 0
        voiced_regions = []
        start_idx = None
        for i, value in enumerate(seg_pitch_values):
            if value > 0:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    voiced_regions.append((start_idx, i))
                    start_idx = None
        if start_idx is not None:
            voiced_regions.append((start_idx, len(seg_pitch_values)))

        # segmenting the audio based on the voiced regions
        segments = []
        spoken_segments_times = []
        for start_idx, end_idx in voiced_regions:
            start_time = seg_pitch_times[start_idx]
            end_time = seg_pitch_times[end_idx - 1]
            duration = end_time - start_time
            if duration < 0.05:
                continue

            # extract pitch values from the global pitch contour
            global_indices = np.where(
                (global_pitch_contour_times >= start_time)
                & (global_pitch_contour_times <= end_time)
            )[0]
            if len(global_indices) == 0:
                continue
            segment_pitch_values = global_pitch_values[global_indices]

            # extract intensity values from the global pitch contour
            intensity_indices = np.where(
                (global_intensity_times >= start_time)
                & (global_intensity_times <= end_time)
            )[0]
            if len(intensity_indices) == 0:
                continue
            segment_intensity = global_intensity_values[intensity_indices]

            segments.append((segment_pitch_values, segment_intensity))
            spoken_segments_times.append((start_time, end_time))

        # Calculate total duration of the audio
        total_duration = sound.duration  # Assuming 'sound' has a 'duration' attribute

        # Calculate silence segments
        silence_segments = []
        previous_end = 0.0
        for start, end in spoken_segments_times:
            if start > previous_end:
                silence_segments.append((previous_end, start))
            previous_end = end
        # Add any remaining silence after the last spoken segment
        if previous_end < total_duration:
            silence_segments.append((previous_end, total_duration))

        # Calculate metrics
        total_spoken_duration = sum(end - start for start, end in spoken_segments_times)
        total_silence_duration = sum(end - start for start, end in silence_segments)
        num_spoken = len(spoken_segments_times)
        num_silence = len(silence_segments)

        # Return all required information
        return (
            segments,  # List of tuples (pitch_values, intensity_values)
            total_spoken_duration,  # Total duration of spoken segments
            total_silence_duration,  # Total duration of silence segments
            num_spoken,  # Number of spoken segments
            num_silence,  # Number of silence segments
        )

    # segment audio in 3 equal parts with the last segment being the whole utterance based on the paper "Timing Levels in Segment-Based Speech Emotion Recognition"
    # Global Relative Time Intervals Approach is used
    def segment_audio(self, audio):

        # calculate the total length of the audio
        audio_data = audio.values[0]
        total_length = len(audio_data)
        segments = []

        # calculate the length of each segment
        segment_length = total_length // 3

        # create 3 segments
        for i in range(3):
            start = i * segment_length
            end = start + segment_length
            segment_data = audio_data[start:end]
            segment_sound = parselmouth.Sound(
                segment_data, sampling_frequency=self.sampling_frequency
            )
            segments.append(segment_sound)

        # add the last segment which is the whole utterance
        segments.append(audio)

        # return (audio/3) + whole audio
        return segments

    def extract_f0(self, segment):
        # F0 extraction based on default values or dynamic pitch floor and ceiling
        method = "default_values"
        if method == "dynamic_values":
            pitch_floor, pitch_ceiling = self.get_boundaries(segment)
        else:
            pitch_floor = self.pitch_floor
            pitch_ceiling = self.pitch_ceiling

        LOGGER.debug(f"Using pitch boundaries: {pitch_floor}-{pitch_ceiling}")

        sound = parselmouth.Sound(segment, sampling_frequency=self.sampling_frequency)
        pitch = sound.to_pitch_ac(
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
            time_step=self.time_step,
            # max_number_of_candidates=4,
            # silence_threshold=0.05,
            # octave_cost=0.01,
            # very_accurate=True,
            # voicing_threshold=0.45
        )
        pitch_values = pitch.selected_array["frequency"]

        return pitch_values

    # based on Speech Prosody: From Acoustics to Interpretation
    def get_boundaries(self, segment):
        sound = parselmouth.Sound(segment)
        # values need to be adjusted
        pitch = sound.to_pitch(pitch_floor=50, pitch_ceiling=300)
        values = pitch.selected_array["frequency"]
        valid_values = values[values > 0]

        q1 = np.percentile(valid_values, 25)
        # print("Q1:", q1)

        pitch_floor = 0.75 * q1
        max_interval = 1.5
        pitch_ceiling = pitch_floor * (2**max_interval)
        # print(pitch_floor, pitch_ceiling)

        return pitch_floor, pitch_ceiling

    def extract_energy(self, segment):
        sound = parselmouth.Sound(segment)
        intensity = sound.to_intensity(
            minimum_pitch=self.pitch_floor, subtract_mean=True, time_step=self.time_step
        )
        intensity_values = intensity.values[0]

        return intensity_values

    def prosody_extraction(self, segment):
        f0 = self.extract_f0(segment)
        energy = self.extract_energy(segment)

        features = {
            "f0": f0,
            "energy": energy,
        }
        return features

    # Aggregation of the features
    class Aggregation:
        def __init__(self):
            self.aggregation_function = {
                "f0": self.aggregate_f0,
                "energy": self.aggregate_energy,
                "duration": self.aggregate_duration,
            }

        def aggregate_duration(
            self, spoken_duration, silence_duration, num_spoken, num_silence
        ):
            stats = []

            num_spoken = float(num_spoken)
            num_silence = float(num_silence)

            avg_spoken = spoken_duration / num_spoken if num_spoken > 0 else 0.0
            avg_silence = silence_duration / num_silence if num_silence > 0 else 0.0
            silence_ratio = num_silence / num_spoken if num_spoken > 0 else 0.0

            stats.extend([avg_spoken, avg_silence, silence_ratio])
            return np.array(stats)

        @staticmethod
        def log_function(value):
            return np.log(value)

        @staticmethod
        def delta_log(value):
            return np.diff(ProsodyEmbedding.Aggregation.log_function(value))

        @staticmethod
        def amplitude_tilt(f0_contour):
            # Nullwerte entfernen
            f0_contour = f0_contour[f0_contour != 0]

            if len(f0_contour) == 0:
                return 0

            # get peak location
            f0_peak_location = np.argmax(f0_contour)

            if f0_peak_location > 0:
                rise_segment = f0_contour[:f0_peak_location]
                rise = (
                    f0_contour[f0_peak_location] - np.min(rise_segment)
                    if len(rise_segment) > 0
                    else 0
                )
            else:
                rise = 0

            fall = f0_contour[f0_peak_location] - np.min(f0_contour[f0_peak_location:])

            if (abs(rise) + abs(fall)) == 0:
                return 0

            # calculate tilt
            tilt = (abs(rise) - abs(fall)) / (abs(rise) + abs(fall))

            return tilt

        @staticmethod
        def duration_tilt(values):

            voiced_indices = np.where(values > 0)[0]
            if len(voiced_indices) == 0:
                return 0.0

            # Bestimme den Index des F0-Peaks in der gesamten Kontur
            peak_location = np.argmax(values)

            rise = peak_location
            fall = (len(values) - 1) - peak_location

            if (rise + fall) == 0:
                return 0.0

            return (rise - fall) / (rise + fall)

        def aggregate_f0(self, values):
            values = np.array(values)
            valid_values = values[values > 0]
            if len(valid_values) == 0:
                return [0, 0, 0, 0, 0]
            f0_tilt = ProsodyEmbedding.Aggregation.amplitude_tilt(values)
            duration_tilt = ProsodyEmbedding.Aggregation.duration_tilt(values)

            try:
                return [
                    np.mean(valid_values),
                    np.max(valid_values),
                    np.max(valid_values) - np.min(valid_values),
                    f0_tilt,
                    duration_tilt,
                ]
            except Exception:
                return [0, 0, 0, 0, 0]

        def aggregate_energy(self, values):
            valid_values = values[values > -60]
            if len(valid_values) == 0:
                return [0, 0, 0, 0, 0, 0]
            try:
                log_values = self.log_function(valid_values)
                delta_log_e = (
                    np.max(log_values) - np.min(log_values)
                    if len(log_values) > 0
                    else 0
                )
                return [
                    np.mean(valid_values),
                    np.max(valid_values),
                    np.min(valid_values),
                    np.std(valid_values),
                    np.mean(log_values),
                    delta_log_e,
                ]
            except Exception:
                return [0, 0, 0, 0, 0, 0]

        # aggregation function of the features
        def aggregation(self, features):
            aggregated_features = []

            for feature_name, value in features.items():
                if feature_name in self.aggregation_function:
                    aggr_value = self.aggregation_function[feature_name](value)
                    aggregated_features.extend(aggr_value)

            return np.array(aggregated_features)

    def create_embedding(self, audio, segmentation_method):
        # choose the segmentation method
        if segmentation_method == "pitch":
            # returns a list of tuples (segmented_pitch, segmented_energy)
            aggregator = self.Aggregation()

            # list to store aggregated pitch values
            f0_aggr_list = []
            energy_aggr_list = []
            (
                segments,
                total_spoken_duration,
                total_silence_duration,
                num_spoken,
                num_silence,
            ) = self.pitch_and_intensity_contour_segmentation_based_on_pitch(audio)

            # iterate over the segments and aggregate the features
            for seg in segments:
                pitch_values, intensity_values = seg
                # aggregate pitch values
                f0_aggr = aggregator.aggregate_f0(np.array(pitch_values))
                # aggregate energy values
                energy_aggr = aggregator.aggregate_energy(np.array(intensity_values))
                f0_aggr_list.append(f0_aggr)
                energy_aggr_list.append(energy_aggr)

            # aggregated features as numpy arrays
            f0_aggr_array = np.array(f0_aggr_list) if f0_aggr_list else np.empty((0, 5))
            energy_aggr_array = (
                np.array(energy_aggr_list) if energy_aggr_list else np.empty((0, 6))
            )

            # calculate global stats with mean
            def compute_global_stats(values):
                # Für jede Feature-Spalte: [mean, std]
                return np.concatenate([np.nanmean(values, axis=0)])

            f0_stats = compute_global_stats(f0_aggr_array)
            energy_stats = compute_global_stats(energy_aggr_array)
            duration_stats = aggregator.aggregate_duration(
                total_spoken_duration, total_silence_duration, num_spoken, num_silence
            )

            # combine aggregated features to final embedding
            final_embedding = np.concatenate([f0_stats, energy_stats, duration_stats])
            final_embedding = np.nan_to_num(final_embedding, nan=0.0)

        elif segmentation_method == "relative":
            segments = self.segment_audio(audio)
            all_features = []
            aggregator = self.Aggregation()

            for i, segment in enumerate(segments):
                features = {}
                for feature_name, segments_to_extract in self.feature_segments.items():
                    if segments_to_extract == "all":
                        features[feature_name] = getattr(
                            self, f"extract_{feature_name}"
                        )(segment)
                    elif segments_to_extract == "first" and i < len(segments) - 1:
                        features[feature_name] = getattr(
                            self, f"extract_{feature_name}"
                        )(segment)
                    elif segments_to_extract == "last" and i == len(segments) - 1:
                        features[feature_name] = getattr(
                            self, f"extract_{feature_name}"
                        )(segment)
                    elif (
                        isinstance(segments_to_extract, list)
                        and i in segments_to_extract
                    ):
                        features[feature_name] = getattr(
                            self, f"extract_{feature_name}"
                        )(segment)

                aggregated = aggregator.aggregation(features)
                all_features.append(aggregated)

            final_embedding = (
                np.concatenate(all_features).flatten() if all_features else np.array([])
            )
        else:
            raise ValueError(f"Unbekannte Segmentierungsmethode: {segmentation_method}")

        return final_embedding
