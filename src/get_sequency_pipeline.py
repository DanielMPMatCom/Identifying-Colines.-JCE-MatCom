import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def get_sequency_pipeline(datapath: str,
    linpath: str = None,
    copath: str = None,
    destination_path_lin: str = None,
    destination_path_co: str = None,
    hop_length: int = -1,
    sample_rate: int = -1,
    plots: bool = True,
    verbose: bool = True,
    attempts: int = 10
) -> None:
    """
    Processes the given data to generate and analyze sequences, dividing them into LIN and CO categories,
    calculating temporal energies, plotting histograms, and building tries.
    
    Parameters:
    - datapath (str): Path to the input data.
    - linpath (str, optional): Path to the LIN data. Defaults to None.
    - copath (str, optional): Path to the CO data. Defaults to None.
    - destination_path_lin (str, optional): Path to save the LIN results. Defaults to None.
    - destination_path_co (str, optional): Path to save the CO results. Defaults to None.
    - hop_length (int, optional): Hop length for plotting sequences. Defaults to -1.
    - sample_rate (int, optional): Sample rate for plotting sequences. Defaults to -1.
    - plots (bool, optional): Whether to plot the sequences. Defaults to True.
    - verbose (bool, optional): Whether to print verbose output. Defaults to True.
    - attempts (int, optional): The number of times the algorithm should be executed. 
    
    Returns:
    None
    """


    if verbose:
        print(
            "---------------------------DIVIDING INTO LIN AND CO--------------------------"
        )
    linpath, copath = divide_into_lin_co(
        datapath=datapath, linpath=linpath, copath=copath
    )

    if verbose:
        print(
            "---------------------------CALCULATING TEMPORAL ENERGIES FOR LIN--------------------------"
        )
    temporal_energies_lin = get_temporal_energies(linpath)

    if verbose:
        print(
            "---------------------------CALCULATING TEMPORAL ENERGIES FOR CO--------------------------"
        )
    temporal_energies_co = get_temporal_energies(copath)

    if verbose:
        print(
            "---------------------------PLOTTING HISTOGRAMS AND GETTING BIN WIDTHS FOR LIN--------------------------"
        )
    limits_lin = plot_energies_histogram_and_get_bin_width(temporal_energies_lin)

    if verbose:
        print(
            "---------------------------PLOTTING HISTOGRAMS AND GETTING BIN WIDTHS FOR CO--------------------------"
        )
    limits_co = plot_energies_histogram_and_get_bin_width(temporal_energies_co)

    if verbose:
        print(
            "---------------------------CALCULATING ENERGY MEANS FOR LIN--------------------------"
        )
    means_lin = get_energy_means(temporal_energies_lin)

    if verbose:
        print(
            "---------------------------CALCULATING ENERGY MEANS FOR CO--------------------------"
        )
    means_co = get_energy_means(temporal_energies_co)

    if verbose:
        print(
            "---------------------------BUILDING TRIES FOR LIN--------------------------"
        )
    tries_lin = build_tries(attempts=attempts, temporal_energies=temporal_energies_lin)

    if verbose:
        print(
            "---------------------------BUILDING TRIES FOR CO--------------------------"
        )
    tries_co = build_tries(attempts=attempts, temporal_energies=temporal_energies_co)

    if verbose:
        print(
            "---------------------------CALCULATING MINIMUN LENGTH--------------------------"
        )
    minimun = min(min_length(tries=tries_lin), min_length(tries=tries_co))

    if verbose:
        print(
            "---------------------------GETTING COLIN SEQUENCE BY ENERGY--------------------------"
        )
    tries_lin, tries_co = colin_sequence_by_energy(
        tries_lin=tries_lin,
        tries_co=tries_co,
        temporal_energies_lin=temporal_energies_lin,
        temporal_energies_co=temporal_energies_co,
        limits_lin=limits_lin,
        limits_co=limits_co,
        minimun=minimun,
        attempts=attempts,
        destination_path_lin=destination_path_lin,
        destination_path_co=destination_path_co,
    )

    if plots:
        if verbose:
            print(
                "---------------------------PLOTTING SEQUENCES--------------------------"
            )
        plot_sequences(
            tries_lin=tries_lin,
            tries_co=tries_co,
            hop_length=hop_length,
            sample_rate=sample_rate,
            attempts=attempts,
            xlim_left=100,
            xlim_right=150,
        )


def divide_into_lin_co(datapath: str, linpath: str = None, copath: str = None) -> tuple:
    """
    Divide .npy files in a directory into two parts and save them in separate directories.

    Parameters:
    - datapath (str): Path to the directory containing .npy files.
    - linpath (str, optional): Path to the directory where the second part of the data will be saved.
                                If not provided, a 'lin' directory will be created inside datapath.
    - copath (str, optional): Path to the directory where the first part of the data will be saved.
                                If not provided, a 'co' directory will be created inside datapath.

    Returns:
    - tuple: A tuple containing the paths to the 'lin' and 'co' directories.

    The function performs the following steps:
    1. Lists all .npy files in the specified datapath.
    2. Creates 'lin' and 'co' directories if they do not exist.
    3. For each .npy file, loads the data, divides it into two parts:
        - The first 60 rows are saved in the 'co' directory.
        - The remaining rows are saved in the 'lin' directory.
    """

    # Obtén la lista de archivos .npy en el directorio
    files = [f for f in os.listdir(datapath) if f.endswith(".npy")]

    if linpath is not None:
        # Crea el directorio 'lin' si no existe
        os.makedirs(linpath, exist_ok=True)
    else:
        linpath = os.path.join(datapath, "lin")
        os.makedirs(linpath, exist_ok=True)

    if copath is not None:
        # Crea el directorio 'co' si no existe
        os.makedirs(copath, exist_ok=True)
    else:
        copath = os.path.join(datapath, "co")
        os.makedirs(copath, exist_ok=True)

    for file in files:
        # Carga el file .npy
        data = np.load(os.path.join(datapath, file))

        # Divide los data en dos partes
        data_lin = data[60:, :]
        data_co = data[:60, :]

        # Guarda las partes en los directorios 'lin' y 'co'
        np.save(os.path.join(linpath, file), data_lin)
        np.save(os.path.join(copath, file), data_co)

    return (linpath, copath)


def get_temporal_energies(files_path: str) -> dict:
    """
    Calculate the temporal energy for each .npy file in a given directory.
    This function lists all .npy files in the specified directory, loads each file as a mel-spectrogram,
    computes the temporal energy by summing the squares of the spectrogram values along the frequency axis,
    and stores the results in a dictionary.

    Args:
        files_path (str): The path to the directory containing .npy files.

    Returns:
        dict: A dictionary where the keys are filenames and the values are arrays of temporal energies.
    """

    files = os.listdir(files_path)  # List all files in the directory

    # Initialize a dictionary to store the temporal energies
    temporal_energies = {}

    # Iterate over each .npy file and calculate the temporal energy for each frame
    for file in files:
        if file.endswith(".npy"):
            # Load the mel-spectrogram
            filepath = os.path.join(files_path, file)
            mel_spectrogram = np.load(filepath)

            # Calculate the temporal energy: sum of squares along the frequency axis (axis 0)
            energy_per_time_frame = np.sum(np.square(mel_spectrogram), axis=0)

            # Add the calculated energy to the dictionary
            temporal_energies[file] = energy_per_time_frame

    # Optional: Display the temporal energies of a specific file
    file_to_show = "20231021_050000a.WAV.npy"
    if file_to_show in temporal_energies:
        print(f"Temporal energies for {file_to_show}:", temporal_energies[file_to_show])

    return temporal_energies


def plot_energies_histogram_and_get_bin_width(temporal_energies: dict) -> list:
    """
    Plots histograms of temporal energies and returns the bin widths.

    Parameters:
    -----------
    temporal_energies : dict
        A dictionary where keys are filenames and values are lists of temporal energies.

    Returns:
    --------
    list
        A list of bin widths for each file.
    """

    # Create a figure with 9 subplots and a size of 16x16 inches
    fig, axs = plt.subplots(3, 3, figsize=(16, 16))

    # Flatten the array of axes to iterate over it
    axs = axs.flatten()

    limits = []

    count = 1
    # Iterate over the files and axes at the same time
    for file, ax in zip(temporal_energies, axs):
        # Calculate the bin width
        bin_width = (max(temporal_energies[file]) - min(temporal_energies[file])) / 50
        limits.append(bin_width)
        print(f"Bin width for {count}: {bin_width}")

        # Create a histogram of the energies for this file
        ax.hist(temporal_energies[file], bins=50)

        # Configure the plot
        ax.set(
            xlabel="Index", ylabel="Frequency", title=f"Energy histogram for {count}"
        )
        ax.set_yscale("log")
        count += 1

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Return the bin widths
    return limits


def get_energy_means(temporal_energies: dict) -> list:
    """
    Calculate the mean energy for each set of temporal energies.
    Args:
        temporal_energies (dict): A dictionary where the keys are identifiers and the values are arrays or lists of energy values.
    Returns:
        list: A list of mean energy values, one for each set of temporal energies.
    """
    means = []
    for energies in temporal_energies.values():
        means.append(energies.mean())

    return means


def build_tries(attempts: int, temporal_energies: dict) -> list:
    """
    Constructs a list of sequences initialized with zeros based on the given number of attempts and temporal energies.

    Args:
        attempts (int): The number of attempts or trials to create.
        temporal_energies (dict): A dictionary where the values are arrays representing temporal energies.

    Returns:
        list: A list of lists, where each inner list contains arrays of zeros with the same shape as the corresponding temporal energy arrays.
    """
    tries = []
    for _ in range(attempts):
        sequences = []
        for energy in temporal_energies.values():
            a = np.zeros(energy.shape[0])
            sequences.append(a)
        tries.append(sequences)
    return tries


def min_length(tries: list) -> int:
    """
    Determines the minimum length among the elements in a 2D list of arrays.

    Args:
        tries (list of list of np.ndarray): A 2D list where each element is a numpy array.

    Returns:
        int: The smallest length found among the arrays in the 2D list.
    """
    minimun = 100000000

    for i in range(len(tries)):
        for j in range(len(tries[0])):
            minimun = (
                minimun if tries[i][j].shape[0] > minimun else tries[i][j].shape[0]
            )
    return minimun


def colin_sequence_by_energy(
    tries_lin: list,
    tries_co: list,
    temporal_energies_lin: dict,
    temporal_energies_co: dict,
    limits_lin: list,
    limits_co: list,
    minimun: int,
    attempts: int,
    destination_path_lin: str = None,
    destination_path_co: str = None,
) -> tuple:
    """
    Executes the Colin sequence algorithm based on energy values for a specified number of attempts.

    Parameters:
    -----------
    tries_lin : list
        A list to store the lin-energy tries for each round.
    tries_co : list
        A list to store the co-energy tries for each round.
    temporal_energies_lin : dict
        A dictionary containing temporal lin-energy values.
    temporal_energies_co : dict
        A dictionary containing temporal co-energy values.
    limits_lin : list
        A list of limit values for lin-energies.
    limits_co : list
        A list of limit values for co-energies.
    minimun : int
        The minimum length to which energy arrays should be truncated.
    attempts : int
        The number of times the algorithm should be executed.
    destination_path_lin : str, optional
        The destination path to save the lin-energy tries. Defaults to None.
    destination_path_co : str, optional
        The destination path to save the co-energy tries. Defaults to None.

    Returns:
    --------
    tuple: tries_lin, tries_co with the stored results
    """

    audios_amount = len(limits_lin)

    # Number of times the algorithm should be executed
    for round in range(attempts):
        mask = np.full(audios_amount, False)
        energycopy_lin = []
        energycopy_co = []

        # Make copies of the energy arrays and truncate them to the same length
        for energy in temporal_energies_lin.values():
            listcopy = energy.copy()
            listcopy = listcopy[:minimun]
            energycopy_lin.append(listcopy)
        for energy in temporal_energies_co.values():
            listcopy = energy.copy()
            listcopy = listcopy[:minimun]
            energycopy_co.append(listcopy)

        # Iterate over the indices of the energy arrays, select the maximum from a random array,
        # copy that value, set it to 0 in the other arrays, and repeat
        for i in range(energycopy_lin[0].shape[0]):
            random = np.random.randint(0, 9)
            while mask[random]:
                random = np.random.randint(0, 9)
            max_lin = energycopy_lin[random].max()
            argmax_lin = energycopy_lin[random].argmax()
            max_co = energycopy_co[random].max()
            argmax_co = energycopy_co[random].argmax()
            if max_lin > limits_lin[random] and max_co > limits_co[random]:
                tries_lin[round][random][argmax_lin] = max_lin
                tries_co[round][random][argmax_co] = max_co
            else:
                mask[random] = True
                if mask.all():
                    break
            for energy2 in energycopy_lin:
                energy2[argmax_lin] = 0
            for energy2 in energycopy_co:
                energy2[argmax_co] = 0

        # Save the tries data to a .npy file
        if destination_path_lin is None:
            destination_path_lin = (
                f"{os.path.dirname(os.path.abspath(__file__))}/tries_lin"
            )
        if destination_path_co is None:
            destination_path_co = (
                f"{os.path.dirname(os.path.abspath(__file__))}/tries_co"
            )

        for j in range(audios_amount):
            # Check if the directory exists, if not, create it
            if not os.path.exists(f"{destination_path_lin}/{round+1}"):
                os.makedirs(f"{destination_path_lin}/{round+1}")
            np.save(
                f"{destination_path_lin}/{round+1}/microphone{j+1}.npy",
                tries_lin[round][j],
            )
            if not os.path.exists(f"{destination_path_co}/{round+1}"):
                os.makedirs(f"{destination_path_co}/{round+1}")
            np.save(
                f"{destination_path_co}/{round+1}/microphone{j+1}.npy",
                tries_co[round][j],
            )

    return tries_lin, tries_co


def plot_sequences(
    tries_lin: list,
    tries_co: list,
    hop_length: int,
    sample_rate: int,
    attempts: int,
    xlim_left: int = -1,
    xlim_right: int = -1,
) -> None:
    """
    Plots sequences of linear and co energy values over time for multiple attempts.

    Parameters:
    - tries_lin (list): A list of lists containing lin-energy values for each attempt.
    - tries_co (list): A list of lists containing co energy values for each attempt.
    - hop_length (int): The hop length used in the audio processing.
    - sample_rate (int): The sample rate of the audio.
    - attempts (int): The number of attempts to plot.
    - xlim_left (int, optional): The left limit for the x-axis. Defaults to -1 (no limit).
    - xlim_right (int, optional): The right limit for the x-axis. Defaults to -1 (no limit).

    Returns:
    - None: This function does not return any value. It displays the plots.

    The function creates a figure with two subplots, one for lin-energy values and one for co energy values.
    It plots the non-zero values of energy over time for each attempt, using different colors for each sequence.
    The y-axis is set to a logarithmic scale, and the x-axis represents time in seconds.
    """

    time_interval = hop_length / sample_rate

    for k in range(attempts):
        arrays_lin = tries_lin[k]
        arrays_co = tries_co[k]

        colors = plt.cm.get_cmap("tab10", 9).colors

        # Create a figure with 2 subplots
        _, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Iterate over the arrays and plot non-zero values
        for i, (array_lin, array_co) in enumerate(zip(arrays_lin, arrays_co)):
            non_zero_indexes_lin = np.where(array_lin > 0)[0]
            non_zero_values_lin = array_lin[non_zero_indexes_lin]

            non_zero_indexes_co = np.where(array_co > 0)[0]
            non_zero_values_co = array_co[non_zero_indexes_co]

            # Convert indexes to milliseconds
            milliseconds_lin = non_zero_indexes_lin * time_interval
            milliseconds_co = non_zero_indexes_co * time_interval

            axs[0].scatter(
                milliseconds_lin,
                non_zero_values_lin,
                color=colors[i],
                label=f"Frog {i+1}",
            )
            axs[1].scatter(
                milliseconds_co,
                non_zero_values_co,
                color=colors[i],
                label=f"Frog {i+1}",
            )

        axs[0].set_title("lin")
        axs[1].set_title("co")

        # Configure the plots
        for ax in axs:
            ax.set_yscale("log")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Energy")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            if xlim_left != -1 and xlim_right == -1:
                ax.set_xlim(
                    xlim_left,
                )
            elif xlim_left != -1 and xlim_right != -1:
                ax.set_xlim(xlim_left, xlim_right)
            elif xlim_left == -1 and xlim_right != -1:
                ax.set_xlim(0, xlim_right)

        plt.suptitle("Sequence")
        # Show the plots
        plt.show()


datapath = "../data/aligned/all"
linpath = "../data/aligned/lin"
copath = "../data/aligned/co"


destination_path_lin = f"../outputs/energy/tries_lin"
destination_path_co = f"../outputs/energy/tries_co"

hop_length = 512
sample_rate = 96000

get_sequency_pipeline(
    datapath=datapath,
    # linpath=linpath,
    # copath=copath,
    destination_path_lin=destination_path_lin,
    destination_path_co=destination_path_co,
    hop_length=hop_length,
    sample_rate=sample_rate,
    plots=True,
    verbose=True,
    attempts=10
)
