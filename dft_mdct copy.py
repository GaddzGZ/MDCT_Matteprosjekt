import numpy as np
import librosa
import scipy.fftpack
import matplotlib.pyplot as plt
import soundfile as sf

def apply_mdct():
    audio_file = librosa.example('trumpet')                 #Laster trompetlyd fra librosa biblioteket
    y, sr = librosa.load(audio_file, sr=None)               #Laster med original sampling rate
    print(f"Loaded {audio_file} with sample rate: {sr} Hz")
   
    #Lagrer den ukomprimerte lydfilen
    sf.write('original_audio.wav', y, sr)
   
    #Definerer verdier
    window_length = 1024  #Vinduslengde
    hop_length = window_length // 1  # 50% overlapp
    total_samples = len(y)
    num_blocks = (total_samples - window_length) // hop_length + 1
    print(f"Total samples: {total_samples}, Number of blocks: {num_blocks}")
   
    # Definerer MDCT-funksjonen
    def mdct(x):
        window = np.sin(np.pi * (np.arange(len(x)) + 0.5) / len(x))  #Bruker sinusvindu
        x_windowed = x * window
        X = scipy.fftpack.dct(x_windowed, type=2, norm='ortho')
        return X[:len(X) // 2]

    #Definerr IMDCT-funksjonen
    def imdct(X):
        N = len(X) * 2  #Ganger med to for å få nok punkter til rekonstruksjon
        window = np.sin(np.pi * (np.arange(N) + 0.5) / N)  #Sinusvindu
        # Utvider X til full lengde, og gjør invers DCT
        X_full = np.zeros(N)
        X_full[:len(X)] = X  #Plasserer MDCT-koeffisienter.
        x = scipy.fftpack.idct(X_full, type=2, norm='ortho') * window
        return x


    #MDCT på hvert ledd
    mdct_results = []
    for start in range(0, len(y) - window_length, hop_length):
        frame = y[start:start + window_length]
        mdct_frame = mdct(frame)
        mdct_results.append(mdct_frame)

    mdct_results = np.array(mdct_results)

    #Komprimerer
    num_coefficients = mdct_results.shape[1]
    threshold = int(num_coefficients * 0.5)  #Halverer antall koeffisienter
    compressed_mdct = np.copy(mdct_results)
    compressed_mdct[:, threshold:] = 0  #Gjør resten av koeffisientene til 0

    print(compressed_mdct)
    #Rekonstruerer lyd med overlapp.
    reconstructed_audio = np.zeros(len(y))
    for i in range(compressed_mdct.shape[0]):
        start = i * hop_length
        frame = imdct(compressed_mdct[i])
        reconstructed_audio[start:start + len(frame)] += frame  #Bruker full lengde av MDCT-output

    # Normalize to avoid clipping
    reconstructed_audio /= np.max(np.abs(reconstructed_audio))

    # Save the compressed audio
    sf.write('compressed_audio.wav', reconstructed_audio, sr)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Original lydfil
    plt.subplot(2, 1, 1)
    plt.title("Original lydfil")
    plt.plot(np.linspace(0, len(y) / sr, len(y)), y, color='blue')
    plt.xlabel("Tid (s)")
    plt.ylabel("Amplitude")
    plt.xlim(0, len(y) / sr)
    

    # Rekonstruert lydfil
    plt.subplot(2, 1, 2)
    plt.title("Rekonstruert lydfil etter MDCT kompresjon")
    plt.plot(np.linspace(0, len(reconstructed_audio) / sr, len(reconstructed_audio)), reconstructed_audio, color='orange')
    plt.xlabel("Tid (s)")
    plt.ylabel("Amplitude")
    plt.xlim(0, len(reconstructed_audio) / sr)

    plt.tight_layout()
    plt.savefig("grafer.svg")
    plt.show()

    return mdct_results, reconstructed_audio

#Kjør funksjonen
mdct_output, reconstructed_audio = apply_mdct()
