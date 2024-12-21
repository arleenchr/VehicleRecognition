# TUGAS 4 Pemrosesan Citra Digital - IF4073
> Tugas 4: Sistem pengenalan jenis kendaraan

## Anggota Kelompok
<table>
    <tr>
        <td>No.</td>
        <td>Nama</td>
        <td>NIM</td>
    </tr>
    <tr>
        <td>1.</td>
        <td>Jason Rivalino</td>
        <td>13521008</td>
    </tr>
    <tr>
        <td>2.</td>
        <td>Arleen Chrysantha Gunardi</td>
        <td>13521059</td>
    </tr>
    <tr>
        <td>3.</td>
        <td>Juan Christopher Santoso</td>
        <td>13521116</td>
    </tr>
</table>

## Table of Contents
* [Deskripsi Singkat](#deskripsi-singkat)
* [Requirements](#requirements)
* [Cara Menjalankan Program](#cara-menjalankan-program)
* [Tampilan GUI Program](#tampilan-gui-program)
* [Pembagian Kerja](#pembagian-kerja)
* [Acknowledgements](#acknowledgements)

## Deskripsi Singkat 
Program yang dibuat dalam tugas ini memiliki kegunaan utama sebagai sistem untuk pengenalan terhadap kendaraan. Terdapat beberapa fitur utama yang diimplementasikan dalam pengerjaan Tugas ini antara lain:
1. Pengenalan kendaraan dengan metode konvensional melalui penggunaan teknik-teknik dalam pengolahan citra
2. Pengenalan kendaraan dengan metode Deep Learning dan CNN

## Requirements
1. Python (<i>default version</i>: 3.10.11)
2. <i>Library</i> pada `requirements.txt`

## Cara Menjalankan Program
Langkah-langkah proses <i>setup</i> program adalah sebagai berikut:
1. <i>Clone</i> repositori ini.
2. [Unduh model CNN](https://drive.google.com/drive/folders/1Rs3EfxcR26MKE-fb0b4-lvCUjBVYG8za?usp=sharing) dan letakkan pada `src/deep_learning/model/cnn_model.pkl`.
3. Jalankan <i>file</i> `main.py` yang terdapat di dalam folder `src` dengan perintah berikut:
```
python src/main.py
```
atau 
```
./run.bat
```
Keterangan: Pastikan sudah meng-<i>install</i> semua <i>library</i> yang dibutuhkan.

## Cara Melatih Model
1. [Unduh <i>file</i> bobot model CNN](https://www.kaggle.com/code/eyadgk/vehicle-rec-via-efficientnet-100-acc-full-guide/output) dan letakkan pada `src/deep_learning/my_model_weights.h5`.
2. Jalankan program `train.py` pada folder `src` dengan perintah berikut:
```
python src/train.py
``` 
atau
```
./train.bat
```

## Tampilan GUI Program
![Screenshot (498)](https://github.com/user-attachments/assets/5a654a70-51e5-4b31-a756-5fd7df7ef373)

## Dataset & Pre-trained Model
- [Dataset 1](https://www.kaggle.com/code/kaggleashwin/dataset-collection-for-vehicle-type-recognition )
- [Dataset 2](https://www.kaggle.com/datasets/rohan300557/vehicle-detection)
- [Pre-trained Model](https://www.kaggle.com/code/eyadgk/vehicle-rec-via-efficientnet-100-acc-full-guide/notebook)

## Pembagian Kerja
<table>
    <tr>
        <td>No.</td>
        <td>Nama</td>
        <td>Kontribusi</td>
    </tr>
    <tr>
        <td>1.</td>
        <td>Jason Rivalino</td>
        <td>
          <ul>
          <li> Pengenalan kendaraan dengan metode Konvensional menggunakan pembelajaran SVM dan KNN
          </ul>
        </td>
    </tr>
    <tr>
        <td>2.</td>
        <td>Arleen Chrysantha Gunardi</td>
        <td>
          <ul>
          <li> Pengenalan kendaraan dengan metode pembelajaran mendalam dengan CNN (EfficientNet)
          </ul>
        </td>
    </tr>
    <tr>
        <td>3.</td>
        <td>Juan Christopher Santoso</td>
        <td>
        <ul>
        <li> Pembuatan GUI Program
        <li> Konfigurasi model Training untuk prediksi kendaraan
        <li> Integrasi GUI dengan sistem program
        </ul>
        </td>
    </tr>
</table>


## Acknowledgements
- Tuhan Yang Maha Esa
- Pak Rinaldi Munir sebagai Dosen Pemrosesan Citra Digital IF4073
