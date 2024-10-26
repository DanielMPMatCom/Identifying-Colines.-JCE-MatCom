import numpy as np
import os


def get_peaks_max_pipeline(
    tries_lin_directory: str,
    tries_co_directory: str,
    destination_path_lin: str,
    destination_path_co: str,
    attempts: int,
    files_amount: int,
    verbose: bool = False,
) -> None:
    """
    Processes and retrieves the maximum peaks from linear and co-linear tries.
    Args:
        tries_lin_directory (str): Directory containing the linear tries.
        tries_co_directory (str): Directory containing the co-linear tries.
        destination_path_lin (str): Path to save the maximum peaks for linear tries.
        destination_path_co (str): Path to save the maximum peaks for co-linear tries.
        attempts (int): Number of attempts to process.
        files_amount (int): Number of files to process.
        verbose (bool, optional): If True, prints detailed processing information. Defaults to False.
    Returns:
        None
    """
    if verbose:
        print("-------------------GETTING TRIES LIN INFO-------------------")
    tries_lin = get_tries(
        tries_directory=tries_lin_directory,
        attempts=attempts,
        files_amount=files_amount,
    )

    if verbose:
        print("-------------------GETTING TRIES CO INFO-------------------")
    tries_co = get_tries(
        tries_directory=tries_co_directory, attempts=attempts, files_amount=files_amount
    )

    if verbose:
        print("-------------------GETTING MAX PEAKS LIN-------------------")
    get_peaks_max(
        destination_path=destination_path_lin,
        attempts=attempts,
        files_amount=files_amount,
        tries=tries_lin,
    )

    if verbose:
        print("-------------------GETTING MAX PEAKS CO-------------------")
    get_peaks_max(
        destination_path=destination_path_co,
        attempts=attempts,
        files_amount=files_amount,
        tries=tries_co,
    )


def get_tries(tries_directory: str, attempts: int, files_amount: int) -> list[list]:
    """
    Loads and returns a list of energy lists from .npy files in a specified directory.

    Args:
        tries_directory (str): The base directory containing subdirectories for each attempt.
        attempts (int): The number of attempts (subdirectories) to process.
        files_amount (int): The number of .npy files to load from each subdirectory.

    Returns:
        list: A list of lists, where each inner list contains the loaded .npy files for a specific attempt.
    """
    tries = []
    for i in range(attempts):
        energylist = []
        directorypath = f"{tries_directory}/{i+1}"
        for j in range(files_amount):
            energylist.append(
                np.load(directorypath + "/microphone" + str(j + 1) + ".npy")
            )
        tries.append(energylist)
    return tries


def justMax(array: np.ndarray) -> None:
    """
    Identifies the maximum value in the given array and sets all other values to 0.

    Parameters:
    array (numpy.ndarray): A 1D numpy array of numerical values.

    Returns:
    None: The function modifies the input array in place.
    """
    current = 0
    currentindex = 0
    for i in range(array.size):
        if array[i]:
            if current < array[i]:
                current = array[i]
                array[currentindex] = 0
                currentindex = i
            else:
                array[i] = 0
        else:
            current = 0
            currentindex = i


def get_peaks_max(
    destination_path: str, attempts: int, files_amount: int, tries: list[list]
) -> None:
    """
    Processes a list of energy values to extract peak values and saves the results to specified files.

    Args:
        destination_path (str): The path where the processed files will be saved.
        attempts (int): The number of attempts or iterations to process.
        files_amount (int): The number of files to be saved per attempt.
        tries (list[list]): A nested list where each sublist contains energy values for each attempt.

    Returns:
        None
    """
    tries_energy = []
    for j in range(attempts):
        energies = []
        for energy in tries[j]:
            listcopy = energy.copy()
            justMax(listcopy)
            energies.append(listcopy)

        tries_energy.append(energies)

        for i in range(files_amount):
            if not os.path.exists(f"{destination_path}/{j+1}"):
                os.makedirs(f"{destination_path}/{j+1}")
            np.save(f"{destination_path}/{j+1}/microphone{i+1}.npy", tries_energy[j][i])


get_peaks_max_pipeline(
    tries_lin_directory="../outputs/energy/tries_lin",
    tries_co_directory="../outputs/energy/tries_co",
    destination_path_lin="../outputs/just_max_peaks/tries_lin",
    destination_path_co="../outputs/just_max_peaks/tries_co",
    attempts=10,
    files_amount=9,
)
