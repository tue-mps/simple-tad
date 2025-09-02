import numpy as np
import torch
import warnings
from typing import Dict, Optional, Sequence, Union, Tuple


class BasicSequencer_Abs:
    def __init__(self, seq_frequency: int, seq_length: Union[int, float]):
        assert seq_frequency > 0, "Sequence frequency must be non-zero and positive!"
        self.seq_frequency = seq_frequency
        if isinstance(seq_length, float) and seq_length > 0:
            self.seq_length = round(seq_length * seq_frequency)
            warnings.warn("Sequence length is given as float, count it in seconds. "
                          f"Given {seq_length} seconds which is {self.seq_length}!")
        elif isinstance(seq_length, int):
            self.seq_length = seq_length
        else:
            raise ValueError(
                "Incorrect seq_length value. It must be float (if in seconds) or int (if in timesteps), "
                f"but given: {seq_length}."
            )

    def get_sequences(self, timesteps_nb: int, input_frequency: int) -> Sequence[Sequence[int]]:
        raise NotImplementedError


class BasicLabeledSequencer_Abs(BasicSequencer_Abs):
    def get_sequences(self, labels: Sequence, input_frequency: int) -> Sequence[Sequence[int]]:
        raise NotImplementedError


class RegularSequencer(BasicSequencer_Abs):
    def __init__(self, seq_frequency: int, seq_length: Union[int, float], step: int = 1):
        super().__init__(seq_frequency=seq_frequency, seq_length=seq_length)
        assert step > 0, f"Step must be at least 1. Given: {step}"
        self.seq_step = step  # step based on seq_frequency

    def get_sequences(self, timesteps_nb: Union[int, Sequence, np.ndarray, torch.Tensor], input_frequency: int) -> Sequence[Sequence[int]]:
        assert input_frequency > 0, f"Input frequency must be positive. Given: {input_frequency}"
        assert input_frequency % self.seq_frequency == 0, \
            ("Cannot convert input frequency to target frequency! Input frequency must be divisible by target "
             f"frequency. Input frequency: {input_frequency}, target frequency: {self.seq_frequency}")
        if isinstance(timesteps_nb, (Sequence, np.ndarray, torch.Tensor)):
            timesteps_nb = len(timesteps_nb)
        # 1. Align fps, skipping some frames. But we also want to keep the last frame. Just because.
        fps_step = input_frequency // self.seq_frequency
        seq_len = fps_step * self.seq_length
        inter_sequence_start = fps_step - 1
        actual_seq_length = seq_len - inter_sequence_start
        if actual_seq_length > timesteps_nb:
            return None
        nb_sequences = int((timesteps_nb - actual_seq_length) // self.seq_step) + 1
        stop = timesteps_nb - actual_seq_length + 1
        start = (timesteps_nb - actual_seq_length) % self.seq_step
        # 3. Get slices of indices to form sequences
        # sequences = [list(range(seq_start_idx, seq_start_idx + actual_seq_length + 1, fps_step))
        sequences = [list(range(seq_start_idx, seq_start_idx+seq_len, fps_step))
                     for seq_start_idx in range(start, stop, self.seq_step)]
        assert all([len(seq) == self.seq_length for seq in sequences]), "Sequences are not of the desired length!"
        assert nb_sequences == len(sequences), "Oops, math has gone wrong! Number of sequences is incorrect!"
        assert sequences[-1][-1] == timesteps_nb - 1, "Oops, math has gone wrong!-2 Help! Beep boop"
        return sequences


class UnsafeOverlapSequencer(BasicLabeledSequencer_Abs):
    """
    This sequencer is for binary classification labels and requires labels to form sequences.
    It takes sequences with a regular step value, but also takes all sequences that end with unsafe label,
    and a few sequences right before and/or after this unsafe sequence. The number of this surrounding
    sequences is indicated by 'surrounding_timesteps' parameter.
        - surrounding_timesteps: None, int, (int,int). How many sequences, surrounding unsafe sequence, to take.
            - None: don't take any surrounding sequences. Identical to (0, 0) or 0.
            - int: take symmetrically <surrounding_timesteps> sequences before AND after.
            - (int, int): take <surrounding_timesteps[0]> before and <surrounding_timesteps[1]> after.
    """
    def __init__(
            self,
            seq_frequency: int,
            seq_length: Union[int, float],
            step: int = 1,
            surrounding_timesteps: Optional[Union[int, Tuple[int, int]]] = None
    ):
        super().__init__(seq_frequency=seq_frequency, seq_length=seq_length)
        assert step > 0, f"Step must be at least 1. Given: {step}"
        self.seq_step = step  # step based on seq_frequency
        if not surrounding_timesteps:
            self.surrounding_timesteps = (0, 0)
        elif isinstance(surrounding_timesteps, int) and surrounding_timesteps >= 0:
            self.surrounding_timesteps = (surrounding_timesteps, surrounding_timesteps)
        elif isinstance(surrounding_timesteps, Sequence) and len(surrounding_timesteps) == 2:
            assert all((isinstance(st, int) and st >= 0) for st in surrounding_timesteps), \
                ("Incorrect format of surrounding_timesteps. It must be a Sequence of ints with length 2! "
                 f"Given: {surrounding_timesteps}")
            self.surrounding_timesteps = surrounding_timesteps

    def get_sequences(self, is_unsafe: Sequence[bool], input_frequency: int) -> Sequence[Sequence[int]]:
        assert input_frequency > 0, f"Input frequency must be positive. Given: {input_frequency}"
        assert input_frequency % self.seq_frequency == 0, \
            ("Cannot convert input frequency to target frequency! Input frequency must be divisible by target "
             f"frequency. Input frequency: {input_frequency}, target frequency: {self.seq_frequency}")
        #
        # 1. Align fps, skipping some frames. But we also want to keep the last frame. Just because.
        timesteps_nb = len(is_unsafe)
        fps_step = input_frequency // self.seq_frequency
        full_seq_step = fps_step * self.seq_step
        full_seq_length = fps_step * self.seq_length
        # 2. Find regular ending sequence positions.
        inter_sequence_start = fps_step - 1
        actual_seq_length = full_seq_length - inter_sequence_start
        stop = timesteps_nb - actual_seq_length + 1
        start = (timesteps_nb - actual_seq_length) % full_seq_step
        nb_sequences = int((timesteps_nb - actual_seq_length) // full_seq_step) + 1
        ends_ids = [seq_start_idx + actual_seq_length - 1 for seq_start_idx in range(start, stop, full_seq_step)]
        assert nb_sequences == len(ends_ids), "Oops, math has gone wrong! Number of sequences is incorrect!"
        # 3. Find unsafe label indices and their surrounding ids
        for idx in range(start + actual_seq_length - 1, timesteps_nb):
            if is_unsafe[idx]:
                idx_from = idx - self.surrounding_timesteps[0] #* fps_step
                idx_to = idx + self.surrounding_timesteps[1] #* fps_step
                ends_ids.extend(range(
                    max(start + actual_seq_length - 1, idx_from),
                    min(timesteps_nb-1, idx_to+1)
                ))
        # 4. Remove duplicates, sort and get final sequences
        ends_ids = sorted(list(set(ends_ids)))
        sequences = [list(range(seq_end_idx - actual_seq_length + 1, seq_end_idx + 1, fps_step)) for seq_end_idx in ends_ids]
        assert all([len(seq) == self.seq_length for seq in sequences]), "Sequences are not of the desired length!"
        assert sequences[-1][-1] == timesteps_nb - 1, "Oops, math has gone wrong!-2 Help! Beep boop"
        return sequences


class RegularSequencerWithStart(BasicSequencer_Abs):
    def __init__(self, seq_frequency: int, seq_length: Union[int, float], step: int = 1):
        super().__init__(seq_frequency=seq_frequency, seq_length=seq_length)
        assert step > 0, f"Step must be at least 1. Given: {step}"
        self.seq_step = step  # step based on seq_frequency

    def get_sequences(self, timesteps_nb: Union[int, Sequence, np.ndarray, torch.Tensor], input_frequency: int) -> Sequence[Sequence[int]]:
        assert input_frequency > 0, f"Input frequency must be positive. Given: {input_frequency}"
        assert input_frequency % self.seq_frequency == 0, \
            ("Cannot convert input frequency to target frequency! Input frequency must be divisible by target "
             f"frequency. Input frequency: {input_frequency}, target frequency: {self.seq_frequency}")
        if isinstance(timesteps_nb, (Sequence, np.ndarray, torch.Tensor)):
            timesteps_nb = len(timesteps_nb)
        # 1. Align fps, skipping some frames. But we also want to keep the last frame. Just because.
        fps_step = input_frequency // self.seq_frequency
        seq_len = fps_step * self.seq_length
        inter_sequence_start = fps_step - 1
        actual_seq_length = seq_len - inter_sequence_start
        if actual_seq_length > timesteps_nb:
            return None
        nb_sequences = int((timesteps_nb - actual_seq_length) // self.seq_step) + 1
        stop = timesteps_nb - actual_seq_length + 1
        start = (timesteps_nb - actual_seq_length) % self.seq_step
        # 3. Get slices of indices to form sequences
        # sequences = [list(range(seq_start_idx, seq_start_idx + actual_seq_length + 1, fps_step))
        sequences = [list(range(seq_start_idx, seq_start_idx+seq_len, fps_step))
                     for seq_start_idx in range(start, stop, self.seq_step)]
        assert all([len(seq) == self.seq_length for seq in sequences]), "Sequences are not of the desired length!"
        assert nb_sequences == len(sequences), "Oops, math has gone wrong! Number of sequences is incorrect!"
        assert sequences[-1][-1] == timesteps_nb - 1, "Oops, math has gone wrong!-2 Help! Beep boop"
        # 4. If the first sequence is too far from the start, add another one
        if start > min(0.3*input_frequency, 5):
            new_seq = list(range(0, 0+seq_len, fps_step))
            assert len(new_seq) == self.seq_length
            sequences.append(new_seq)
        return sequences