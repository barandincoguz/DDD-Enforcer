# Hocayla Konuşulacaklar — Gündem Notları

> Hocaya tezimle ilgili sunulması/onaylanması gereken konuların listesi.
> Her konu altında: ricam, gerekçe, iş yükü, karşılık ve sonraki adım.

---

## 1. Bağımsız Validator — TEDU'daki Diğer Hoca'dan Rica

### Yazar Ekibi (referans)
**Yazarlar (3)**: Baran, Ali, Murat Karakaya (supervisor)
**Rater ekibi (Fleiss's κ)**: Baran + Ali + **TEDU bağımsız Hoca'sı (external rater)**
**Murat Karakaya rater DEĞİL** — supervisor/yazar olarak kalır; "supervisor double as rater" probleminden kaçınır.

### Asıl Süreç
Sen TEDU bağımsız Hoca'ya direkt yaklaşacaksın (zaten tanıyorsun). Murat Hoca'ya bu rica HABER niteliğinde paylaşılır (onay sorulmaz, ama courtesy olarak bilgilendirilir, çünkü tezim çalışması).

### Murat Hocaya Söyleyecek Cümle (kısa, akademik nezaket)
**"Hocam, tezim için bağımsız bir akademisyen incelemesi gerek. TEDU'da [diğer Hoca'nın adı]'na yaklaşacağım — projeden bihaber olduğu için ideal. Sizden onay istemiyorum, ama bilginiz olsun istedim."**

### Neden Bağımsız Validator Lazım
Paper'ımızda LLM tool'umuzun kararlarını başka bir LLM (Judge) ile puanlıyoruz. Springer Empirical Software Engineering dergisinin reviewer'ları büyük olasılıkla şunu soracak:

> *"Yapay zeka ile yapay zekanın işini kontrol ediyorsunuz; bir insan bunu doğruladı mı?"*

Bağımsız bir akademisyenin küçük bir örneklem üzerinde "AI'ın kararına katılıyor musunuz?" diye işaretlemesi, bu soruyu kapatan tek geçerli yöntem. LLM-merkezli ESE çalışmaları için topluluk standardı haline gelen bir adım (G5 — Use Human Validation for LLM Outputs).

### Diğer Hoca'nın İş Yükü
- **1 saat eğitim**: Ben kahveye davet edip "DDD-Enforcer ne yapıyor, ben senden ne istiyorum" diye 1-sayfalık özet üzerinden anlatırım.
- **12–15 saat asıl iş**: Bir Excel/Google Sheet'te ~150 vakalı bir tabloda satır-satır gidip "AI'ın kararına katılıyor musunuz?" diye ☑/☒ işaretleme.
- **3 hafta yayılabilir**, sıkı deadline yok — kendi vaktinde, ilgili olduğunda yapar.
- Asıl çalışma penceresi: tezimin **8–10. haftaları arası** (Temmuz-Ağustos 2026 civarı).

### Karşılığında Ne Sunuyoruz
1. **Paper'da Acknowledgement** (zorunlu): *"We thank Prof. [Adı] for independent validation."*
2. **Replication package'a erken erişim**: dataset'imizi kendi araştırmalarında kullanabilir.
3. **Future-collaboration ihtimali** (opsiyonel): bu data üzerinde kendi makalesini yapmak isterse beraber çalışmaya açığız.
4. **Co-authorship** (sadece Hoca'mın insiyatifinde): eğer audit sırasında metodoloji'mize ciddi içerik katkısı verirse 3. yazar olabilir; ama bu vaadi şu aşamada vermiyoruz.

### Sonraki Adımlar
1. **Bu hafta**: Murat Hoca'ya kısa bilgi notu (yukarıdaki cümle) — courtesy
2. **Aynı hafta veya sonrası**: TEDU bağımsız Hoca'ya direkt yaklaş (öğrenciden öğretmene rica formatı)
3. **Onay alınınca**: Ben 1-sayfalık "DDD-Enforcer brifingi" + örnek 5 vakalı mini-template hazırlarım. Bu pakete bakıp Hoca "evet" derse calibration session schedule edilir.
4. **8. hafta**: Asıl audit başlar, 150 vaka

### Karar Alanı / Doldur
- [ ] TEDU bağımsız Hoca'nın adı/unvanı: ____________________
- [ ] Murat Hoca'ya bilgilendirildi mi: ☐ Evet  ☐ Hayır
- [ ] Bağımsız Hoca'ya yaklaşma tarihi: ____________________
- [ ] Bağımsız Hoca'nın ilk tepkisi: ____________________
- [ ] Bir sonraki adım: ____________________

---

## 2. RQ5'in Kapsamdan Çıkarılması — Bilgilendirme

### Asıl Konu (Hocaya Bildirim)
**"Hocam, çalışma haritası üzerinden tekrar geçtim. Sizin '5 araştırma sorusu' notunuz vardı; iki yazar olmamız ve 14 haftalık takvimimiz göz önüne alındığında, RQ5'i scope'tan çıkarmaya karar verdim. RQ1–RQ4'e derinlik versek daha güçlü bir çıktı olur."**

### Neden RQ5'i Düşürdük
1. **Kapasite**: 2 yazarız (planın v1'inde 3 yazar varsayılıyordu). 18 work-package + 5 RQ + 14 hafta → realistik değil.
2. **Rigor önceliği**: RQ5'i de yetiştirmeye çalışsak istatistiksel güç (Wilcoxon, Friedman+Nemenyi, Cliff's δ), expert validation (D3), ve replication package için harcayacağımız zaman azalır → reviewer'lar burayı yakalar.
3. **Risk yönetimi**: 4 RQ'da metodolojik derinlik, 5 RQ'da yüzeysellikten daha güçlü bir paper çıkarır. Empirical Software Engineering dergisi "depth over breadth" prensibini tercih ediyor.

### Reviewer'lar Bu Karardan Haberdar Olacak Mı?
**Hayır.** RQ5 paper draft'ında resmen yer almamıştı, sadece bizim iç planımızdaydı. Cover letter'da bahsetmeyeceğiz. Reviewer'lar 4 RQ'lu temiz bir paper görecek; geriye dönük bir scope-cut imajı oluşmayacak.

### Eğer Reviewer "Neden 5+ Domain veya Daha Fazla Çalışma Yok?" Sorarsa
Bu zaten paper'da "future work" başlığı altında karşılanacak (örn. "scaling to 8-10 domains, ablation studies, practitioner field study" gibi maddelerle). Detayını ileride Konu olarak `HOCA_GUNDEM.md`'ye eklerim.

### Hocaya Onay Sorulacak Mı?
**Bilgilendirme niteliğinde**, onay sormuyoruz; ben karar verdim. Ama Hoca'nın güçlü itirazı varsa konuşmaya açığız.

### Karar Alanı / Doldur
- [ ] Hocaya bildirildi mi: ☐ Evet  ☐ Hayır
- [ ] Hocan'ın tepkisi: ____________________
- [ ] İtiraz varsa nasıl karşılanır: ____________________

---

## 3. RQ1 Pipeline Sayısı — Onay/Görüş

### Soru
**"Hocam, RQ1'de 2 pipeline mı (saf single-call vs multi-agent) yoksa 3 pipeline mı (naïve + RAG + multi-agent) karşılaştıralım? Şu anki plan 3 pipeline. Sizin görüşünüz nedir?"**

### Mevcut Plan: 3 Pipeline (P1 + P2 + P3)
- **P1**: Saf single-call (SRS + kod tek prompt'ta gider, LLM çıktıyı verir)
- **P2**: Retrieval-augmented single-call (SRS embeddings'e gömülür, kod analiz ederken ilgili pasajlar çekilir)
- **P3**: Multi-agent (Scout → Architect → Specialist → Synthesizer)

### Neden 3 Pipeline Önerim (kısa)
1. **RAG, 2026'da SE topluluğunun "default baseline"i** — RAG yokluğu reviewer'ın ilk soracağı şey ("why not retrieval?")
2. **Üç-katmanlı ablation = empirik derinlik**: P1→P2 farkı = retrieval'ın katkısı, P2→P3 farkı = agent mimari katkısı. Marjinal değer ayrıştırılır.
3. **Compute maliyeti ihmal edilebilir**: 30 vs 20 koşum (1-1.5 saat fark). RQ2-4'e zaten sadece kazanan pipeline taşınıyor, çarpan etkisi yok.
4. **Mevcut altyapı zaten var**: `chromadb` + `sentence-transformers` requirements.txt'de, paper draft P1/P2/P3 yapısıyla yazılmış. P2'yi atmak iş geri-atmak olur.
5. **Confound izolasyonu**: P1 vs P3 farkı "prompt uzunluğu mu, multi-step mi, retrieval mı?" karışık. P2 ortak prompt-length sağlar, retrieval'ın etkisini izole eder.

### 2 Pipeline'ın Tek Riski
RAG implementasyonu zayıfsa (kötü embedding, irrelevant retrieval) "bilerek zayıf strawman" gibi görünür. Faz 2'de smoke-test ile mitigate edilir.

### Hocadan Beklediğim Görüş
- **Onaylıyorsa**: Plan üzerinde değişiklik yok, RQ1 = 3 pipeline.
- **2 pipeline tercih ederse**: Gerekçesini sor (örn. "scope ağır, RAG'i future work'e at"), tartış. Onun gerekçesi kabul edilebilirse paper draft + WP-01d güncellemesi gerek (~1-2 gün iş).

### Karar Alanı / Doldur
- [ ] Hoca'nın görüşü: ☐ 3-pipeline OK  ☐ 2-pipeline tercih  ☐ Başka öneri: ____________________
- [ ] Görüşme tarihi: ____________________
- [ ] Final karar: ____________________

---

<!-- Aşağı yeni konular eklenecek; örn: D2 SRS seçim onayı, future-work çerçevesi, vs. -->
