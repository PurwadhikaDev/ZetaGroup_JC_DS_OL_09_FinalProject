Final Project Zeta Group - Bank Marketing Campaigns: Prediction of Bank Marketing Campaign Success
-------------------------
Created By:
1. Benaya Aprian Lisias
2. Angga Gustama Putra
3. Christianto Kurniawan Priyono

Business Problem Understanding
------------------------------------------------------------------------------------
**Context**
Berdasarkan data Portuguese Banking Association, diketahui bahwa industri perbankan Portugal mengalami peningkatan saldo deposito. Hal ini membuat rasio pinjaman terhadap deposito atau biasa disebut dengan LDR (Loan to Deposit Ratio) mengalami penurunan. Dikutip dari Stephen Buschbom dalam Assessing Bank Risk Using the Loan-to-Deposit Ratio, diketahui bahwa LDR sering digunakan sebagai indikator tingkat risiko bank, dimana rasio yang tinggi menunjukkan bahwa bank mengambil lebih banyak risiko karena memiliki cadangan kas yang lebih sedikit untuk menutupi kerugian yang tidak terduga.

Data peforma industri Bank Portugal ini menyoroti kondisi yang menguntungkan dalam sektor industri perbankan karena LDR-nya yang mengalami penurunan, hal ini menunjukkan bahwa bank-bank Portugal telah memperkuat posisi likuiditas-nya sehingga level risikonya menjadi lebih rendah dari sebelumnya dan dapat memainkan peran penting dalam mendukung perekonomian. Hal ini memungkinkan bank-bank untuk meningkatkan kemampuan mereka dalam menyediakan pinjaman kepada individu, bisnis, dan sektor-sektor lain dalam perekonomian. Ketersediaan likuiditas yang cukup dapat membantu dalam mendorong aktivitas ekonomi yang sehat, seperti investasi, konsumsi, dan pertumbuhan bisnis.

Hal ini menunjukkan bahwa deposito merupakan salah satu sumber dana yang penting bagi bank dalam menjalankan operasionalnya. Oleh karena itu, kampanye pemasaran yang menargetkan peningkatan jumlah deposito dapat memberikan manfaat besar bagi bank.

Mempertimbangkan pertumbuhan deposito yang terus berlanjut, tentunya risiko penurunan deposito akan tetap ada. Oleh karena itu penting bagi bank untuk memprediksi apakah nasabah akan melakukan deposito atau tidak. Prediksi ini sangat penting karena berpotensi berdampak pada keseluruhan operasional dan keberhasilan bank dalam memitigasi risiko hal yang tak terduga.

**Problem Statement** <br>
Dikutip dari berbagai sumber pada expatica, diketahui ada lebih dari 150 bank di Portugal. Meskipun jumlah cabang bank fisik di Portugal menurun selama dekade terakhir, masih ada sekitar 32,8 cabang per 100.000 penduduk. Dimana itu hampir tiga kali lipat rata-rata global. Tentunya hal ini menjadikan persaingan industri perbankan di Portugal semakin kompetitif. Untuk menjawab tantangan tersebut, maka penting bagi tim marketing untuk dapat mengalokasikan sumber daya dengan lebih efisien agar dapat memperoleh hasil yang optimal.

Dari uraian diatas, diketahui bahwa masalah yang perlu dianalisa adalah "Pelanggan seperti apa yang harus tim marketing bank targetkan untuk memaksimalkan perolehan deposito?"

Berdasarkan standar quality management ISO 9001: 2015 pada klausa 6, perusahaan harus dapat mengambil tindakan untuk mengidentifikasi risiko dan peluang, serta merencanakan cara menangani risiko dan peluang yang teridentifikasi. Tentunya dalam konteks ini bank perlu secara proaktif mengelola risiko kemungkinan penurunan deposito untuk menjaga LDR yang stabil dan memastikan likuiditas yang cukup. Sebagai bentuk mitigas, bank harus dapat memprediksi kemungkinan nasabah akan melakukan deposito atau tidak.

Maka dapat diketahui bahwa masalah lain yang perlu dianalisa adalah "Bagaimana bank (khususnya tim management risiko) dapat memprediksi nasabah akan melakukan deposito atau tidak untuk menjaga stabilitas LDR (Loan to Deposit Ratio)?"

Target:
0 : Nasabah tidak melakukan deposito
1 : Nasabah melakukan deposito

**Goals**
Sebagaimana dalam menghadapi tantangan tersebut, maka project ini memiliki tujuan untuk dapat memberikan rekomendasi dengan memanfaatkan machine learning kepada stakeholder sebagai berikut:

1. Tim marketing perlu memprediksi kemungkinan seorang nasabah dalam merespon kampanye pemasaran deposito. Sehingga tim marketing dapat mengoptimalkan biaya pemasaran dengan lebih efisien.
2. Dalam upaya mempertahankan peforma bank, tim management risiko perlu memprediksi kemunkinan nasabah akan melakukan deposito atau tidak. Sehingga top management dapat membuat strategi kebijagan yang matang lebih awal.
3. Berkaitan dalam analisa masalah tersebut, maka perlu dianalisa untuk mengetahui faktor / variable apa yang membuat seorang nasabah melakukan deposito pada bank. Sehingga tim marketing dan management risiko dapat membuat rencana yang lebih baik dalam pendekatan ke potensial nasabah yang melakukan deposito dan memitigasi kemungkinan jumlah deposito yang turun

**Analytics Approach**
Jadi yang akan dilakukan dalam project ini adalah menganalisis data untuk menemukan pola yang membedakan nasabah yang melakukan deposito dan yang tidak melakukan deposito.

Kemudian dalam project ini akan membangun model klasifikasi yang akan membantu tim marketing bank untuk dapat memprediksi probabilitas seorang nasabah yang akan/ingin melakukan deposito di bank tersebut atau tidak.

**Metric Evaluation**
	        | N-Prediction |  P-Prediction |
--- | --- | --- |
N-Actual  | TN	         |  FP |
P-Actual  | FN	         |  TP |

FN (False Negative) : Model salah memprediksi bahwa seorang nasabah tidak akan melakukan deposito, padahal sebenarnya nasabah tersebut akan melakukan deposito
--> Konsekuensi : Bank kehilangan peluang untuk menarik nasabah potensial yang sebenarnya akan merespons positif terhadap kampanye.

FP (False Positive) : Model salah memprediksi bahwa seorang nasabah akan melakukan deposito, padahal sebenarnya nasabah tidak akan melakukan deposito
--> Konsekuensi : Bank mengalokasikan sumber daya tambahan kepada nasabah yang sebenarnya tidak berminat

TN (True Negative) : Seorang nasabah tidak akan melakukan deposito dan sebenarnya nasabah tersebut memang tidak akan melakukan deposito
--> Manfaat : Bank menghindari pengeluaran yang tidak perlu dan menghemat sumber daya untuk nasabah yang tidak berminat atau tidak akan merespons positif

FP (True Positive) : Seorang nasabah akan melakukan deposito dan sebenarnya nasabah tersebut memang akan melakukan deposito.
--> Manfaat : Bank berhasil menarik nasabah yang berpotensi merespons positif dan berkontribusi pada pendapatan dan pertumbuhan bisnis.

Berdasarkan tujuan dari project untuk memprediksi sebanyak - banyaknya nasabah mana yang akan melakukan deposito (TP) dan yang tidak (TN), maka metric utama yang akan digunakan adalah **accuracy**.

Accuracy = TN+TP / TP+TB+FP+FN


Data Understanding
------------------------------------------------------------------------------------
**Attribute Information**

Dataset source: https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset

| Section                      | Attribute        | Type        | Description                                                                                                                                                               |
|------------------------------|------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bank Client Data             | age              | Numerical   | Umur nasabah                                                                                                                                                             |
| Bank Client Data             | job              | Categorical | Tipe pekerjaan nasabah       |
| Bank Client Data             | marital          | Categorical | Status Pernikahan                                                                     |
| Bank Client Data             | education        | Categorical | Jenjang Edukasi                                                 |
| Bank Client Data             | default          | Categorical | Status kredit terganggu                                                                                                                                |
| Bank Client Data             | housing          | Categorical | Pinjaman untuk keperluan rumah                                                                                                                                 |
| Bank Client Data             | loan             | Categorical | Pinjaman untuk keperluan personal                                                                                                                                |
| Campaign Data                | contact          | Categorical | Tipe kontak yang digunakan                                                                                                                     |
| Campaign Data                | month            | Categorical | Bulan terakhir nasabah dikontak                                                                                                     |
| Campaign Data                | day_of_week      | Categorical | Hari terakhir nasabah dikontak                                                                                                            |
| Campaign Data                | duration         | Numerical   | Jumlah detik durasi nasabah dikontak                                      |
| Campaign Data                | campaign         | Numerical   | Jumlah kontak yang dilakukan selama kampanye                                                                             |
| Campaign Data                | pdays            | Numerical   | Jumlah hari berlalu dari nasabah terakhir dikontak saat kampanye, 999 artinya nasabah sebelumnya tidak dikontak                              |
| Campaign Data                | previous         | Numerical   | Jumlah kontak yang dilakukan sebelum kampanye                                                                                                     |
| Campaign Data                | poutcome         | Categorical | Hasil dari kampanye sebelumnya (previous)                                                                                         |
| Social and Economic Context  | emp.var.rate     | Numerical   | Tingkat Variasi Pekerjaan - indikator triwulanan, menunjukkan apakah pasar kerja berkembang atau berkontraksi. Nilai positif menunjukkan peningkatan kesempatan kerja, sedangkan nilai negatif menunjukkan penurunan                                                                                                                          |
| Social and Economic Context  | cons.price.idx   | Numerical   | Indeks Harga Konsumen - indikator bulanan, perubahan harga barang dan jasa umum yang biasanya dikonsumsi rumah tangga, seperti makanan, perumahan, transportasi, dan perawatan kesehatan                                                                                                                             |
| Social and Economic Context  | cons.conf.idx    | Numerical   | Indeks Kepercayaan Konsumen - indikator bulanan, menunjukkan seberapa percaya diri konsumen tentang ekonomi dan prospek keuangan pribadi mereka                                                                                                                           |
| Social and Economic Context  | euribor3m        | Numerical   | Tarif 3 Bulan Euribor - indikator triwulanan, biaya pinjaman untuk bank-bank di zona euro selama periode tiga bulan                                                                                                                                    |
| Social and Economic Context  | nr.employed      | Numerical   | Jumlah karyawan - indikator triwulan, total pekerja yang bekerja dalam suatu wilayah atau negara pada suatu periode waktu tertentu                                                                                                                               |
| Output variable (desired target) | y              | Categorical | Nasabah melakukan deposito ( yes/no)                                                                                                                  |

Dataset Overview
1. Terdapat 41188 baris dan 21 kolom yang memberikan informasi tentang marketing bank
2. Setiap baris data merepresentasikan informasi dari hasil tim marketing menawarkan seorang nasabah untuk melakukan deposito
3. Fitur bersifat categorical dan numerical (Nominal, Ordinal, Binary), beberapa dengan kardinalitas tinggi
4. Dataset tidak seimbang untuk variable target (y)


Exploratory Data Analysis (EDA)
------------------------------------------------------------------------------------
Langkah-langkah yang dilakukan dalam EDA adalah: 
1. Checking Data Duplicate
2. Checking Value In Dataset
3. Checking 'Unknown' Value Distribution
4. Checking Data Proportion
5. Checking Numerical Data Distribution
6. Checking COrrelation
7. EDA Summary


Preprocessing
------------------------------------------------------------------------------------
Langkah-langkah yang dilakukan dalam preprocessing adalah: 
1. Handling Data Duplicate
2. Handling Inconsistent Value
3. Handling Binary Value (Default, Housing, Loan)
4. Handling 'Unknown' Value (Job, Marital, Education, Default, Housing, Loan)
5. Handling Outlier (Pdays & Previous)

Data Analysis
------------------------------------------------------------------------------------
**A. Latar Belakang Nasabah**
Dalam upaya memprediksi nasabah melakukan deposito atau tidak, diperlukan analisa lebih lanjut untuk mengetahui latar belakang nasabah seperti apa yang melakukan deposito. Berkaitan dengan hal tersebut, maka diketahui insight sebagai berikut:
   1. Diketahui bahwa fitur (age, job, marital, education) dapat dijadikan referensi untuk mengetahui nasabah melakukan deposito atau tidak.
   2. Diharapkan tim marketing dapat membuat strategi pemasaran sesuai dengan preferensi segmen nasabah yang minimal memiliki salah satu kategori berikut
      a. age (10s, 60s s/d 90s karena persentase melakukan deposit tinggi). Strategi dapat menonjolkan manfaat jangka panjang dari menabung dan berinvestasi sejak dini untuk segmen 10s dan menciptakan rasa urgensi. Kelompok usia 60-an hingga 90-an dapat berfokus pada keamanan finansial, perencanaan pensiun, dan pembangunan warisan.
      b. job (admin & blue-colar karena potensi jumlahnya yang banyak). Strategi dapat membuat kampanye atau insentif yang memprioritaskan grup karyawan seperti admin & blue-colar seperti opsi akun deposito yang disederhanakan dan ramah pengguna yang mudah dipahami dan diakses
      c. Karena faktorisasi menunjukkan bahwa kelompok nasabah 'retired' memiliki pengaruh signifikan terhadap perolehan deposito, buat kampanye pemasaran yang dirancang khusus untuk pensiunan. Sorot stabilitas dan keamanan produk deposito, bersama dengan fitur atau manfaat khusus yang memenuhi kebutuhan mereka. Pertimbangkan bermitra dengan organisasi atau klub pensiunan untuk meningkatkan visibilitas dalam demografis ini.
      d. Karena nasabah dengan status 'married' memiliki jumlah deposito terbesar dan cukup dominan, fokuskan upaya pemasaran pada grup ini. Buat kampanye yang menekankan tujuan keuangan bersama, seperti menabung untuk rumah, pendidikan, atau pensiun. Tonjolkan manfaat produk deposito dalam membantu mereka mencapai aspirasi bersama. Pertimbangkan untuk menawarkan promosi atau insentif khusus yang disesuaikan dengan pasangan suami istri.
      e. Peluang lain yang dapat dioptimalkan adalah membuat program marketing untuk fokus meningkatkan jumlah nasabah dengan status 'single' karena persentasenya untuk melakukan deposito lebih tinggi
      f. education (university karena jumlah dan persentase melakukan deposit tinggi). Strategi dapat memposisikan bank sebagai penasihat tepercaya yang diharapkan dapat membantu menarik dan mempertahankan deposito mereka.
   3. Diharapkan tim risk management dapat berkolaborasi dengan tim marketing dalam penyesuaian tidak hanya melakukan pendekatan ke segmen tersebut agar tidak ketergantungan karena dapat membahayakan perusahaan. Tim risk management juga harus mempertimbangkan mitigasi nasabah yang memiliki 'unknown'value karena ketidakpastian identifikasi background mereka
   4. Fitur (housing, loan, default) memiliki korelasi yang lemah dalam keputusan nasabah melakukan deposito (kolom y), namun fitur ini tetap dapat dipertimbangkan sebagai tambahan fitur model dalam memprediksi keputusan nasabah melakukan deposito.
**B. Marketing Campaign**
Proses yang perlu dipertimbangkan dalam memprediksi nasabah akan melakukan deposito adalah analisa marketing campaign. Maka akan dilakukan analisa untuk menjawab pertanyaan berikut.
   1. Diketahui bahwa fitur (contact, month, poutcome) dapat dijadikan referensi untuk mengetahui nasabah melakukan deposito atau tidak.
   2. Fitur day_of_week dapat dipertimbangkan untuk dimasukan dalam model.
   3. Diharapkan tim marketing dapat membuat strategi pemasaran dengan melakukan:
    a. Alokasikan sumber daya dan upaya untuk strategi pemasaran seluler. Kembangkan kampanye ramah seluler, optimalkan situs web untuk pengguna seluler, dan manfaatkan aplikasi SMS atau perpesanan untuk terlibat dengan pelanggan.
    b. Fokuskan upaya pemasaran pada bulan Maret, Desember, dan September, karena bulan-bulan tersebut telah menunjukkan tingkat penerimaan yang lebih tinggi untuk nasabah
    c. Sesuaikan penawaran promosi, insentif, dan pengiriman pesan agar selaras dengan karakteristik khusus bulan-bulan lain yang memiliki tingkat penerimaan sedikit
    d. Sederhanakan proses penawaran, membuatnya lebih informatif, mudah digunakan, dan efisien untuk memanfaatkan durasi yang terbatas
   4. Tim risk mangement dapat menyusun terkait mitigasi berikut:
    a. Penilaian risiko menyeluruh terhadap aktivitas pemasaran seluler, dengan mempertimbangkan potensi kerentanan keamanan, dan masalah privasi data
    b. Persiapkan rencana darurat untuk mengatasi setiap lonjakan permintaan, memastikan sumber daya dan infrastruktur yang memadai untuk menangani permintaan dan transaksi pelanggan yang meningkat
**C. Social & Economic Context**
Faktor sosial dan ekonomi mungkin dapat memengaruhi selera risiko individu, yang memengaruhi keputusan mereka terkait penempatan deposito. Maka akan dilakukan analisa untuk menjawab pertanyaan berikut.
   1. Dari analisa korelasi yang diberikan menunjukkan bahwa faktor-faktor seperti index harga konsumen yang lebih tinggi (cons.conf.idx), pasar kerja yang memburuk (emp.var.rate), suku bunga yang lebih rendah (euribor3m), dan pasar kerja yang melemah (nr.employed) terkait dengan kemungkinan penurunan nasabah membuat deposit. Namun perlu dipertimbangkan bahwa korelasi tersebut lemah, maka diperlukan fitur tambahan dalam model machine learning nantinya.


Methodology
------------------------------------------------------------------------------------
  1. **Data Splitting**
      Pada splitting, kita menggunakan 20% test size dan stratify=y. fungsi stratify digunakan untuk memastikan bahwa distribusi kelas dalam variabel target tetap seimbang antara set pelatihan dan pengujian.
  2. **Encoding**
      Dilakukan encoding dengan beberapa metode untuk feature categorical.
      a. Onehot Encoding untuk kolom marital, contact, dan poutcome
      b. Binary Encoding untuk kolom job, education, month, dan day_of_week
  3. **Modeling**
      Pada modeling, kita menggunakan classifier : LogisticRegression, KNeighborsClassifier, RandomForestClassifier, DecisionTreeClassifier, AdaBoostClassifier, GradientBoostingClassifier, XGBClassifier. Pada Imbalance: RandomOverSampler, SMOTE, dan RandomUnderSamper. dan Scaler MinMaxScaler, StandardScaler, RobustScaler.
      
      Dengan hasil, Best Scaler: MinMaxScaler, Best Imbalance: None, Best Model: GradientBoostingClassifier, dengan Best Score: 0.9156344869459623
  4. **Feature Selection**
      Pada proses ini, kita mempersempit feature yang akan digunakan menjadi 30% feature total.
  5. **Hyperparameter Tuning**
      Hyperparameter Tuning kali ini menggunakan teknik GridSearchCV. Teknik ini digunakan untuk mencari nilai parameter yang optimal dari dataset yang diberikan dalam bentuk grid.
  6. **Feature Importance**
      Pada teknik ini, digunakan untuk mencari feature importance pada dataset.


Conclusion dan Recommendation
------------------------------------------------------------------------------------
**Conclusion**
| | Accuracy with Train | Accuracy with Test | True Negative | True Positive | False Negative | False Positive 
| --- | --- | --- |
| First Model | 92% | 92% | 7063 | 489 | 439 | 245
| Model Feature with Feature Selection | 92% | 92% | 7071 | 474 | 454 | 237
| Model Feature with Feature Selection + Tuning | 92% | 92% | 7077 | 493 | 435 | 231
| | Same % between train and test is better | More is better | Less is better

Diketahui bahwa model dengan feature selection + tuning memiliki performa yang lebih baik dari segala aspek, maka model tersebut adalah model final yang akan digunakan.

**Recommendation**
For Model: 
1. Melakukan tuning parameter yang lebih banyak pada model yang digunakan. Dengan mengeksplorasi berbagai kombinasi parameter, dapat meningkatkan performa dan akurasi model prediksi.
2. Penambahan fitur baru seperti waktu nasabah mendaftar di bank yang memungkinkan melihat hubungan lamanya nasabah menabung dan keputusan deposit
3. Tim marketing dapat mengushakan perolehan data yang lengkap tanpa nilai 'unknown' untuk meminimalisir interpetasi terutama pada kolom default

For Team Marketing:
Berkaitan dengan pengaruh duration yang sangat kuat dalam keputusan pelanggan melakukan deposit, tim marketing perlu lebih dalam untuk menganalisis perjalanan pelanggan dari titik kontak awal hingga selesai kontak. Identifikasi potensi kendala atau area di mana pelanggan berhenti selama proses berlangsung. Proses kontak pelanggan yang lancar dan efisien dapat meningkatkan tingkat konversi dan mendorong lebih banyak pelanggan untuk melakukan deposit sehingga perolehan deposito lebih optimal.

For Team Risk Management: 
Secara teratur memantau kondisi ekonomi termasuk tingkat penduduk yang memiliki pekerjaan dan tingkat 3 bulan Euribor. Identifikasi tren dan perubahan undustri dan suku bunga serta potensi dampaknya terhadap perilaku pelanggan

Hal ini untuk membuat keputusan yang tepat dan mengembangkan strategi yang selaras dengan kondisi ekonomi yang berlaku untuk mempertahanka performa LDR (Loan to Deposit)
