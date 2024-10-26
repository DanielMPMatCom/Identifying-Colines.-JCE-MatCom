import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import pandas as pd


from src import (
    align_audios_pipeline,
    get_frequency_graph_pipeline,
    get_mel_spectrograms,
    get_peaks_max_pipeline,
    get_sequency_pipeline,
)


def main_pipeline(
    input_directory: str,
    output_directory: str,
    attempts: int = 10,
    files_amount: int = 9,
    verbose: bool = False,
    plots: bool = False,
):

    hop_length = 512
    n_fft = 6096
    n_mels = 128
    f_min = 1600
    f_max = 4096
    sr = 96000
    block_duration = 10

    if verbose:
        print(
            "-------------------GETTING MEL SPECTROGRAMS FROM AUDIOS-------------------"
        )
    get_mel_spectrograms.get_mel_spectrogram_from_audio(
        input_directory=input_directory,
        output_directory=output_directory,
        block_duration=block_duration,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )

    not_aligned_files_directory = output_directory

    if plots:
        if verbose:
            print("-------------------PLOTTING MEL SPECTROGRAMS-------------------")
        get_mel_spectrograms.plot_mel_spectrogram_graph_from_matrix(
            data_source_path=not_aligned_files_directory,
            vmin_energy_scale=0,
            vmax_energy_scale=1,
            xlim_left=0,
            xlim_right=1000,
        )

    pivot_file_name = "20231021_190000h.npy"
    preprocessed_data_path = "../data/not_aligned/freq_cut"
    data_destination_path = "../data/aligned/all"

    if verbose:
        print("-------------------ALIGNING AUDIOS-------------------")
    aligned_audios = align_audios_pipeline.align_audios_pipeline(
        data_source_path=not_aligned_files_directory,
        pivot_file_name=pivot_file_name,
        preprocessed_data_path=preprocessed_data_path,
        data_destination_path=data_destination_path,
        plots=plots,
        verbose=verbose,
    )

    aligned_files_directory = data_destination_path

    linpath = "../data/aligned/lin"
    copath = "../data/aligned/co"

    destination_path_lin = f"../outputs/energy/tries_lin"
    destination_path_co = f"../outputs/energy/tries_co"

    if verbose:
        print("----------------------GETTING SEQUENCIES----------------------")
    get_sequency_pipeline.get_sequency_pipeline(
        datapath=aligned_files_directory,
        linpath=linpath,
        copath=copath,
        destination_path_lin=destination_path_lin,
        destination_path_co=destination_path_co,
        hop_length=hop_length,
        sample_rate=sr,
        plots=plots,
        verbose=verbose,
        attempts=attempts,
    )

    just_max_peaks_diretory_lin = "../outputs/just_max_peaks/tries_lin"
    just_max_peaks_diretory_co = "../outputs/just_max_peaks/tries_co"

    if verbose:
        print("----------------------GETTING MAX PEAKS----------------------")
    get_peaks_max_pipeline.get_peaks_max_pipeline(
        tries_lin_directory=destination_path_lin,
        tries_co_directory=destination_path_co,
        destination_path_lin=just_max_peaks_diretory_lin,
        destination_path_co=just_max_peaks_diretory_co,
        attempts=attempts,
        files_amount=files_amount,
    )

    data_destination_path_lin = "../outputs/frequency/tries_lin"
    data_destination_path_co = "../outputs/frequency/tries_co"

    if verbose:
        print("----------------------GETTING FREQUENCY GRAPH----------------------")
    get_frequency_graph_pipeline.get_frequency_graph_pipeline(
        just_max_peaks_diretory_lin=just_max_peaks_diretory_lin,
        just_max_peaks_diretory_co=just_max_peaks_diretory_co,
        attempts=attempts,
        files_amount=files_amount,
        lin_mel_spectrograms_directory=linpath,
        co_mel_spectrograms_directory=copath,
        data_destination_path_lin=data_destination_path_lin,
        data_destination_path_co=data_destination_path_co,
        verbose=verbose,
        plots=plots,
    )


attempts = 10
files_amount = 9
input_directory = "../data/not_aligned/audios"
output_directory = "../data/not_aligned/all"


# Example of use
main_pipeline(
    input_directory=input_directory,
    output_directory=output_directory,
    attempts=attempts,
    files_amount=files_amount,
    verbose=True,
    plots=True,
)
