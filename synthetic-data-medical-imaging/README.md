# Synthetic Data in Medical Image Analysis: A Critical Evaluation

**Dylan Gelbier**, MSc Artificial Intelligence, Loughborough University, 2026

*Adapted from coursework for COP500 Research Methods (January 2026).*

📄 [Download as PDF](Synthetic_Data_in_Medical_Image_Analysis.pdf)

---

## Contents

- [Introduction](#introduction)
- [Part 1: Mapping the Field](#part-1-mapping-the-field)
  - [1.1 Research Map](#11-research-map)
  - [1.2 Medical Image Context](#12-medical-image-context)
  - [1.3 Synthetic Data Paradigm](#13-synthetic-data-paradigm)
  - [1.4 Integration into the Medical Imaging Pipeline](#14-integration-into-the-medical-imaging-pipeline)
  - [1.5 Evaluation Strategies](#15-evaluation-strategies)
  - [1.6 Validation Assumptions and Limitations](#16-validation-assumptions-and-limitations)
  - [1.7 Open Challenges and Research Gaps](#17-open-challenges-and-research-gaps)
  - [Summary of Part 1](#summary-of-part-1)
- [Part 2: Two Contrasting Studies](#part-2-two-contrasting-studies)
  - [2.1 Positioning the Papers](#21-positioning-the-papers)
  - [2.2 Critique of Paper A: Gao et al. (2023)](#22-critique-of-paper-a-gao-et-al-2023)
  - [2.3 Critique of Paper B: Al Khalil et al. (2023)](#23-critique-of-paper-b-al-khalil-et-al-2023)
  - [2.4 Compare and Contrast](#24-compare-and-contrast)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

Medical image analysis is a core component of modern clinical decision-making (Litjens et al., 2017), supporting tasks such as detection, segmentation and registration for diagnosis and prognosis (Hussain et al., 2022). Despite significant advances in deep learning methods, progress in this area remains constrained by the availability, quality and governance of clinical imaging data. These challenges are particularly important in safety-critical medical settings, where regulatory and ethical constraints restrict data sharing and reuse (Ranschaert et al., 2021).

Generating synthetic medical data has strong potential to address these issues, offering a way to mitigate data scarcity, class imbalance and privacy concerns without relying directly on sensitive patient information. Synthetic data has been widely proposed (Goncalves et al., 2020) across multiple domains; despite this, its adoption in medical fields has been slow and hesitant. This hesitation reflects concerns around realism, bias and trustworthiness, each of which places stricter demands on data validity than in non-medical domains (Frangi et al., 2018).

While synthetic data is often evaluated through performance gains, growing evidence suggests that accuracy alone is insufficient for assessing clinical validity (Alaa et al., 2022). Although multiple evaluation approaches beyond performance metrics have been proposed, there is no widely accepted or consistently applied framework for evaluating synthetic data in medical image analysis.

This article explores how synthetic data is generated, integrated and evaluated within medical image analysis pipelines, with particular attention to modality- and task-specific constraints, current limitations and open research challenges.

---

## Part 1: Mapping the Field

### 1.1 Research Map

![Radial research map. The central node, "Synthetic Data in Medical Image Analysis", branches into six areas, each with sub-topics: Validation Assumptions + Limitations (generalisation failures, lack of external validation, overfitting to benchmarks, dataset-specific conclusions, fragmented evaluation practices); Evaluation Practices (mismatch between reported performance and clinical trust, clinical validity, bias and fairness evaluation, data fidelity, performance-centric metrics such as accuracy and DICE); Integration into Medical Image Pipeline (pre-training vs fine-tuning, training-time augmentation, data balancing with rare class synthesis, domain adaptation); Synthetic Data Paradigm (generation-based, simulation-based, augmentation-based approaches and hybrid pipelines); Medical Image Context (clinical constraints; clinical tasks: detection, segmentation, registration, prognostics, diagnosis; imaging modalities: X-ray, PET, CT, MRI); and Open Challenges + Research Gaps (lack of standardised validation frameworks, weak linkage between performance and trust, poor reproducibility across datasets and modalities).](figures/figure-1-research-map.png)

***Figure 1:** Research map outlining the structure of the field of synthetic data in medical image analysis.*

Figure 1 presents a research territory map outlining the structure of the field of synthetic data in medical image analysis. The diagram captures the major conceptual areas identified through the literature review, including clinical context, synthetic data paradigms, integration within learning pipelines, and evaluation and validation considerations. It provides a structured overview of how these components relate and interact within contemporary research, with the aim of piecing together how synthetic data is being received in medical image analysis.

### 1.2 Medical Image Context

The application of machine learning to medical image analysis is shaped by the clinical context in which imaging data is acquired and used. This imposes constraints that do not typically arise in non-medical domains.

#### Clinical Tasks in Medical Image Analysis

Before assessing the relevance of synthesising data, attention must be drawn to the importance of medical images in clinical tasks such as detection, segmentation, registration and diagnosis. Each is a broad category, but all are central to the decisions and actions of medical experts (Litjens et al., 2017); the identification of cancerous lesions illustrates the gravity involved. Although these all fall under the umbrella of clinical tasks that rely on medical imaging, important differences must be noted: each requires different label types, tolerates different levels of error and imposes different realism requirements on data.

#### Imaging Modalities and Physical Constraints

Goceri (2023) highlights the importance of distinguishing between imaging modalities. Medical experts rely on a wide range of imaging types, including MRI, CT, PET and X-ray. Each modality differs in acquisition physics, image quality (resolution, contrast, etc.), noise characteristics and typical artefacts. Every image holds many characteristics that make it unique and difficult to replicate; the realism of synthetic data is therefore modality-dependent.

#### Safety-Critical and Regulatory Context

Medical images carry inherent risk because they support clinical decisions, and errors can have direct consequences for patients. This is a strong reason why data governance is stricter than in other domains (Jiang et al., 2025). Limited data sharing and an increased reliance on small datasets both stem from these constraints, and both motivate interest in synthetic data.

#### Implications for Synthetic Data

Synthetic data has been proposed as a solution for the medical imaging pipeline, raising the ceiling for medical deep learning projects. Nevertheless, because of task diversity, modality specificity and the large clinical risk, synthetic data cannot be treated as a simple, generic augmentation; each decision must meet task- and modality-specific requirements. Its use must therefore be evaluated more cautiously.

### 1.3 Synthetic Data Paradigm

#### Defining Synthetic Data in Medical Image Analysis

Put simply, synthetic data can be defined as artificially generated or modified data intended to supplement or replace real clinical data (Goncalves et al., 2020).

#### Augmentation-Based Approaches

The most common paradigm is data augmentation (Goceri, 2023), which covers traditional transformations such as rotation, scaling and intensity changes. These modifications can be seen as task-preserving and are widely accepted clinically due to their simplicity and low risk of producing unrealistic samples. Caveats include its limited diversity, which is often insufficient for rare classes, and its inability to create new anatomical structures; its importance is task-dependent but remains high.

#### Simulation-Based Approaches

Often used for modality-specific cases, simulation generates data through physics-informed modelling (Frangi et al., 2018). It offers high interpretability and control over parameters, hence its usefulness for domain adaptation. However, development costs are high, and simplified assumptions may fail to capture real-world variability, which is an objective in medical image synthesis.

#### Generation-Based Approaches

Generation-based methods, including GANs and diffusion models, learn to synthesise images directly from real data distributions. This approach is attractive because the resulting images have high visual realism (Shin et al., 2018). Nevertheless, because of bias replication, memorisation risk and the difficulty of evaluating fidelity, the medical field's perfectionism makes it hesitant.

#### Hybrid Pipelines

Some pipelines combine simulation, generation and augmentation (Goncalves et al., 2020; Frangi et al., 2018). Arguably, hybrid approaches reflect the limitations of individual paradigms rather than their maturity.

#### Comparative Perspective and Limitations

No paradigm is universally optimal (Goncalves et al., 2020); the choice should depend on the task, modality and data (Goceri, 2023). Despite the diversity of available paradigms, the literature lacks clear guidance on how to select appropriate synthetic data strategies for specific medical imaging contexts.

### 1.4 Integration into the Medical Imaging Pipeline

#### Synthetic Data as a Pipeline Component

Performance gains often arise from pipeline design choices rather than from synthetic data alone. Its impact depends heavily on when and how synthetic data is used (Gao et al., 2023).

#### Training-Time Augmentation

Combining synthetic and real data during training has been shown to improve model performance (Shin et al., 2018). However, the amount of synthetic data used appears to be set case by case rather than standardised.

#### Data Balancing and Rare Class Synthesis

Synthetic data is frequently used to address class imbalance (Shin et al., 2018; Litjens et al., 2017), which is very common in the medical field, as seen in pathology and anomaly detection; it is regarded as a strong way to improve sensitivity to rare conditions. However, the risks of oversimplifying rare cases and of synthetic artefacts dominating reveal its limitations.

#### Pre-Training and Fine-Tuning Strategies

In some pipelines, models are pre-trained on large synthetic datasets before being fine-tuned on smaller real clinical datasets, a form of transfer learning (Li et al., 2023). This strategy can improve convergence and feature learning, especially when real data is scarce. However, its effectiveness depends on the alignment between synthetic and real data distributions, and gains may diminish when synthetic pre-training introduces biased representations.

#### Domain Adaptation and Generalisation

Synthetic data is also used to support domain adaptation, for example when models trained on one scanner, institution or modality are applied elsewhere. While synthetic variation can improve robustness to distribution shifts, it may also encode institutional biases, limiting true generalisation across clinical settings (Yu et al., 2022).

#### Critical Synthesis and Limitations

Across studies, the integration strategy determines the observed benefit, but many studies under-report pipeline details (Yu et al., 2022), so reported gains may not generalise beyond specific set-ups. Consequently, improvements attributed to synthetic data must be interpreted in light of the surrounding pipeline design and validation strategy.

### 1.5 Evaluation Strategies

While synthetic data generation has advanced rapidly, evaluation practices have lagged, often relying on indirect and task-dependent proxies.

#### Dominance of Performance-Centric Evaluation

Most existing studies evaluate synthetic data indirectly by measuring performance (accuracy, DICE or AUC) rather than the properties of the data itself.

#### What Performance Metrics Fail to Capture

Performance metrics offer limited visibility and can hide underlying problems; improvements can coexist with reduced robustness, poor generalisation and unrealistic artificial features. Problems arise when metrics such as accuracy are applied to synthetic data, as they may not detect bias replication (Jiang et al., 2025) and say little about clinical plausibility. Frangi et al. (2018) addresses cases where pixel-level similarity does not equate to usefulness.

#### Data Fidelity and Statistical Similarity Metrics

Several studies have proposed statistical similarity metrics (Alaa et al., 2022); these approaches aim to quantify how closely synthetic data resembles real clinical data at the pixel and feature level. These metrics address how closely synthetic data matches real data, but high statistical similarity does not guarantee that it will be clinically valid.

#### Clinical Validity and Expert-Centred Evaluation

Automated metrics may fail to capture the diagnostic realism and anatomical coherence of images (Frangi et al., 2018; Goceri, 2023); this has led several studies to incorporate expert assessment of the clinical plausibility of synthetic images.

#### Fragmentation and Lack of Standardised Evaluation Frameworks

Taken together, existing evaluation strategies remain fragmented, with little consensus on how to validate the use of synthetic data. The absence of a standardised evaluation framework limits the interpretability and reproducibility of reported performance gains. As a result, assessing the trustworthiness of synthetic data in medical image analysis remains an unresolved challenge.

### 1.6 Validation Assumptions and Limitations

One must read this literature critically: much of the analysis relies on implicit validation assumptions that are not explicitly addressed. One such assumption is that statistical similarity is sufficient evidence of clinical validity; this is a faulty premise when generative models are trained on biased datasets.

A further assumption is that performance gains with synthetic data indicate clinical robustness; however, improvements in benchmark results may reflect overfitting rather than improved generalisation to unseen clinical environments. Clinically, this is concerning, as relying on such models in safety-critical settings is risky. Rankin et al. (2020) shows such discrepancies between results derived from synthetic and real data.

Finally, there is an underlying assumption that synthetic data reduces the ethical and privacy risks associated with real data (Rankin et al., 2020). While synthetic data may remove direct identifiers, questions remain regarding data leakage (Mendelevitch & Lesh, 2021) and the possibility that sensitive data is encoded and regenerated; these uncertainties complicate claims that synthetic data provides a universally safer alternative to real clinical datasets.

Collectively, these assumptions highlight fundamental limitations in current validation practices and emphasise the need for cautious interpretation of results involving synthetic data in medical image analysis.

### 1.7 Open Challenges and Research Gaps

It is important to identify specific absences within this research, as they reveal the field's immaturity. The following outlines the main gaps found.

#### Lack of Standardised Evaluation Frameworks

A fundamental challenge in this field is the absence of standardised evaluation frameworks, as highlighted by Alaa et al. (2022). Across tasks, modalities and clinical contexts, studies use incompatible metrics, which makes results difficult to compare. There also appears to be a heavy reliance on an illogical, accuracy-dominated evaluation culture.

#### Weak Connection between Synthetic Data Performance and Clinical Trust

Another significant gap is the large disconnect between medical professionals and ML researchers, which hinders clinical trust (Yu et al., 2022). Improved model performance does not translate directly into clinical adoption (Yu et al., 2022); this may be because clinicians evaluate safety differently from ML researchers. A culture of distrust appears to have grown, given that synthetic data is rarely tested in real workflows or even evaluated in clinical decision-making contexts.

#### Limited External Validation

Most deep learning studies in medical imaging remain only internally validated (Yu et al., 2022). As medical imaging varies considerably across devices, protocols and populations, synthetic data may overfit to institutional biases, which reduces willingness to adopt medical image synthesis.

#### Unresolved Question of Fully Synthetic Datasets

A critical unresolved question in the field is whether fully synthetic datasets can meaningfully replace real clinical data or whether some reliance on real data is unavoidable. Gao et al. (2023) appears optimistic about this potential. Research shows a risk of compounding biases, as synthetic data often depends on real data for calibration and generation (Jiang et al., 2025); that being said, the benefits of privacy and scalability should make full synthetic data the goal. Currently, most approaches still use real data, and there is evidence that performance may plateau without it (Shin et al., 2018; Nowruzi et al., 2019). Persistent doubt and the small scale of existing trials leave the question open.

#### Regulatory, Ethical and Governance Uncertainty

Finally, the regulatory and ethical status of synthetic medical imaging data remains uncertain, which poses a large barrier to widespread clinical adoption (Jiang et al., 2025). Image synthesis still has imperfections: studies show that it can leak patient information (Jiang et al., 2025), so it cannot yet be considered fully ethical, particularly given widespread concern about real-data privacy and the consensus for change. In addition, there is considerable legal ambiguity caused by a lack of regulations and standards; without a clear consent model, this uncertainty will continue.

### Summary of Part 1

Part 1 has examined the research landscape of synthetic data in medical image analysis and how it sits in the broader context of clinical imaging pipelines. The analysis reveals a rapidly expanding yet structurally fragmented research area. The research map (Figure 1) reflects the topic's structure, breaking it down into medical context, synthetic data paradigms, pipeline integration strategies, evaluation practices and unresolved challenges. The literature reveals that synthetic medical data is promising, with great potential to mitigate data scarcity and privacy constraints; however, governance problems, a lack of standardised validation and clinical distrust leave a large gap.

---

## Part 2: Two Contrasting Studies

### 2.1 Positioning the Papers

The scope of synthetic data in medical image analysis has expanded rapidly in response to persistent challenges such as data scarcity, privacy constraints and limited generalisability. This part spotlights two papers, Gao et al. (2023) and Al Khalil et al. (2023), which together illustrate the breadth of the medical synthetic data research paradigm: one emphasises performance and scalability via simulation, while the other foregrounds robustness and clinical variation. Gao et al. (2023) adopts a simulation-driven approach, exploring whether large-scale synthetically generated X-ray datasets can produce models that generalise effectively on real tasks. In contrast, Al Khalil et al. (2023) investigates how GAN-based synthetic augmentation influences deep learning models across clinically variable MRI data. Together, they reflect two active debates in the literature: the potential of synthetic data as a substitute for real clinical images, and its value as an enhancing complement to real data under diverse conditions.

Gao et al. (2023) focuses on the challenges of data scarcity and generalisation; conceptually, this paper aligns with work that seeks to diminish reliance on limited real clinical data by expanding training data through controlled simulation. However, there is an implicit assumption that metric performance gains directly correspond to improved clinical generalisation, which points to a broader methodological tension in the field: performance metrics often serve as stand-ins for clinical validity, despite their known limitations.

Al Khalil et al. (2023) situates itself in a complementary but distinct position by emphasising the robustness of deep learning segmentation models when using generative imagery, proposing that synthetic images can supplement existing datasets to improve model adaptability. Although it shows that synthetic images can largely stand in for real ones, it presents synthetic data mainly as a tool for strengthening performance in realistic scenarios alongside real data.

These two studies illustrate a spectrum of perspectives on the role of synthetic data in medical imaging. At one end, simulation-based methods, like that in Gao et al. (2023), emphasise scalability through synthetic training sets. At the other, augmentation-focused methods, as in Al Khalil et al. (2023), prioritise robustness to clinical variation. This conceptual contrast captures an ongoing shift in the field; researchers are increasingly aware that synthetic data must not only achieve performance gains in controlled evaluations but also engage with clinical realities such as population and hardware diversity.

### 2.2 Critique of Paper A: Gao et al. (2023)

Gao et al. (2023) focuses on data scarcity, privacy constraints and scalability in medical imaging. Before critiquing it, it is important to state explicitly that Paper A does address these issues: the large synthetic dataset enables systematic experimentation at a scale that would be infeasible with real patient data. It also claims that its simulation method offers controlled generation with precise, automatically generated ground truth. The paper is published in a peer-reviewed journal, indicating a high standard of methodological review and relevance to the field, and it engages extensively with the existing literature.

Nevertheless, it is important to highlight its limitations. Fundamentally, the paper evaluates success almost entirely through quantitative task performance. Therefore, there is an implicit assumption that improved accuracy equates to clinical usefulness, which has direct implications for interpretability, trust and real-world applicability.

Continuing this critique of clinical usefulness, the paper appears to lack expert-led validation. Medical imaging is safety-critical, and the absence of clinician-supported evaluation hinders future clinical adoption.

There is also a strong dependence on simulation assumptions, which limits the model's generalisability because synthetic images are deeply affected by simulator design choices. This invites scepticism, as real-world complexity may not be fully captured, limiting robustness to unseen clinical variability.

### 2.3 Critique of Paper B: Al Khalil et al. (2023)

Al Khalil et al. (2023) positions synthetic data mainly as a complement to real clinical datasets, focusing on GAN-based synthesis to improve model robustness.

It must be stated that Paper B selected a segmentation-informed conditional GAN, designed to produce semantically coherent synthetic MRI images; given the results reported, this choice could be considered successful. Despite this, the paper offers only a brief rationale for this type of GAN and never compares it with alternatives. Many other types of GAN exist, so why were none of them assessed and reported? Data synthesis needs to mirror complex, variable tissue structure; should future work in this field simply assume that a segmentation-informed variant is always the right choice? This missing comparison is a significant limitation.

The paper's narrow scope (a single region, the heart; a single segmentation task; and a single modality, MRI) suggests limited generalisability. The reader is left questioning whether the model is transferable to other imaging tasks, organs and modalities.

Although GAN-based augmentation improves models by providing more visually realistic data, the clinical validity of the generated structures is not systematically assessed; beyond one check of derived ejection fraction, the synthetic images receive no expert evaluation. One cannot assume that visual realism equates to diagnostic realism. Are metrics such as the DICE score and Hausdorff distance sufficient proxies for clinical trustworthiness? They could be, but that must be established through rigorous testing, which unfortunately was not provided. Given the limited expert evaluation and the absence of structured clinical scoring, this is a clear limitation.

### 2.4 Compare and Contrast

Although both studies aim to improve medical image analysis through synthetic data, they explore the problem in different ways. At the core of both papers, the researchers are trying to solve data scarcity in a highly privacy-sensitive field. Exploring the domain is difficult with such high ethical standards; each move should be calculated and justified. Each paper tries to find a distinct solution. Gao et al. (2023) emphasises scalability and dataset expansion through simulation, hailing it as a great solution and amplifying the positives of its performance scores, whereas Al Khalil et al. (2023) prioritises robustness enhancement through targeted augmentation. This contrast highlights a broader methodological divide in the field.

Methodologically, Gao et al. (2023) relies on controlled physical assumptions to enable large-scale data generation, with the aim of purely synthetic training; in contrast, Al Khalil et al. (2023) relies on GAN-based generation embedded within existing real datasets. This is a fundamental divide: one paper sees synthetic data as having the potential to be an ethical, purely synthetic alternative, while the other sees it mainly as an aid to fill gaps in real data.

However, this ambition risks rushed solutions: the success of the results in Gao et al. (2023) can easily be attributed to performance generalisation, which cannot be said to be clinically grounded. Al Khalil et al.'s (2023) GAN proposition seems to carry more clinical weight, potentially owing to its focus on segmentation robustness and data fidelity. Therefore, it could be said to have more practical integration potential.

The differences in approach extend across scale, generation type, evaluation focus and risk profile. Collectively, these papers suggest that synthetic data is unlikely to function as a universal replacement for real clinical data in the near term; instead, its most reliable value currently lies in domain-specific robustness enhancement. This variety underlines the need for an evaluation framework that extends beyond performance metrics.

---

## Conclusion

To conclude, both papers provide excellent insights into, and additions to, the field of synthetic medical imaging, suggesting that synthetic data has an important place in the medical industry in addressing the problems that affect its data.

Reflecting this, it is apparent that synthetic data in medical image analysis cannot be assessed through performance improvements alone but must be understood in relation to clinical context. By contrasting the simulation-based scalability with augmentation-driven robustness, this article has revealed the tension between technical performance and clinical reliability that currently shapes the field. When evaluating any topic in a field as safety-critical as the medical field, there should always be an emphasis on the clinical environment – everything from clinician-approved validation methods to the deployment setting – and medical reasoning should always take priority. Despite this, a disconnect appears to remain.

Each paper brings together a plethora of ideas; the comparison of Gao et al. (2023) and Al Khalil et al. (2023) suggests that the role of synthetic data in medical imaging is more nuanced than initially assumed. There isn't a universal solution to data scarcity, but each synthetic data solution introduces trade-offs between realism and scalability. The proposal of different methodologies for different modalities offers researchers and clinicians an exciting opportunity for creative exploration in solving their medical image data issues. Both papers report promising results that present synthetic data as a real solution. However, future success in this research can only stem from standardisation, governance and a push for clinical partnership; without these, the field risks becoming stuck.

---

## References

Alaa, A., Van Breugel, B., Saveliev, E.S. and Van Der Schaar, M., 2022, June. How faithful is your synthetic data? Sample-level metrics for evaluating and auditing generative models. In *International Conference on Machine Learning* (pp. 290–306). PMLR. <https://proceedings.mlr.press/v162/alaa22a.html>

Al Khalil, Y., Amirrajab, S., Lorenz, C., Weese, J., Pluim, J.P.W. and Breeuwer, M., 2023. On the usability of synthetic data for improving the robustness of deep learning-based segmentation of cardiac magnetic resonance images. *Medical Image Analysis*, 84, p.102688. <https://doi.org/10.1016/j.media.2022.102688>


Frangi, A.F., Tsaftaris, S.A. and Prince, J.L., 2018. Simulation and synthesis in medical imaging. *IEEE Transactions on Medical Imaging*, 37(3), pp.673–679. <https://doi.org/10.1109/TMI.2018.2800298>

Gao, C., Killeen, B.D., Hu, Y., Grupp, R.B., Taylor, R.H., Armand, M. and Unberath, M., 2023. Synthetic data accelerates the development of generalizable learning-based algorithms for X-ray image analysis. *Nature Machine Intelligence*, 5(3), pp.294–308. <https://doi.org/10.1038/s42256-023-00629-1>

Goceri, E., 2023. Medical image data augmentation: techniques, comparisons and interpretations. *Artificial Intelligence Review*, 56(11), pp.12561–12605. <https://doi.org/10.1007/s10462-023-10453-z>

Goncalves, A., Ray, P., Soper, B., Stevens, J., Coyle, L. and Sales, A.P., 2020. Generation and evaluation of synthetic patient data. *BMC Medical Research Methodology*, 20(1), p.108. <https://doi.org/10.1186/s12874-020-00977-1>

Hussain, S., Mubeen, I., Ullah, N., Shah, S.S.U.D., Khan, B.A., Zahoor, M., Ullah, R., Khan, F.A. and Sultan, M.A., 2022. Modern diagnostic imaging technique applications and risk factors in the medical field: a review. *BioMed Research International*, 2022(1), p.5164970. <https://doi.org/10.1155/2022/5164970>

Jiang, J., Domingues, L. and Mendes, J.M., 2025. Synthetic data in medical imaging within the EHDS: a path forward for ethics, regulation, and standards. *Frontiers in Digital Health*, 7, p.1620270. <https://doi.org/10.3389/fdgth.2025.1620270>

Li, M., Jiang, Y., Zhang, Y. and Zhu, H., 2023. Medical image analysis using deep learning algorithms. *Frontiers in Public Health*, 11, p.1273253. <https://doi.org/10.3389/fpubh.2023.1273253>

Litjens, G., Kooi, T., Bejnordi, B.E., Setio, A.A.A., Ciompi, F., Ghafoorian, M., Van Der Laak, J.A., Van Ginneken, B. and Sánchez, C.I., 2017. A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, pp.60–88. <https://doi.org/10.1016/j.media.2017.07.005>

Mendelevitch, O. and Lesh, M.D., 2021. Fidelity and privacy of synthetic medical data. *arXiv preprint* arXiv:2101.08658. <https://arxiv.org/abs/2101.08658>

Nowruzi, F.E., Kapoor, P., Kolhatkar, D., Hassanat, F.A., Laganiere, R. and Rebut, J., 2019. How much real data do we actually need: analyzing object detection performance using synthetic and real data. *arXiv preprint* arXiv:1907.07061. <https://arxiv.org/abs/1907.07061>

Rankin, D., Black, M., Bond, R., Wallace, J., Mulvenna, M. and Epelde, G., 2020. Reliability of supervised machine learning using synthetic data in health care: model to preserve privacy for data sharing. *JMIR Medical Informatics*, 8(7), p.e18910. <https://doi.org/10.2196/18910>

Ranschaert, E., Topff, L. and Pianykh, O., 2021. Optimization of radiology workflow with artificial intelligence. *Radiologic Clinics of North America*, 59(6), pp.955–966. <https://doi.org/10.1016/j.rcl.2021.06.006>

Shin, H.C., Tenenholtz, N.A., Rogers, J.K., Schwarz, C.G., Senjem, M.L., Gunter, J.L., Andriole, K.P. and Michalski, M., 2018, September. Medical image synthesis for data augmentation and anonymization using generative adversarial networks. In *International Workshop on Simulation and Synthesis in Medical Imaging* (pp. 1–11). Cham: Springer International Publishing. <https://doi.org/10.1007/978-3-030-00536-8_1>

Yu, A.C., Mohajer, B. and Eng, J., 2022. External validation of deep learning algorithms for radiologic diagnosis: a systematic review. *Radiology: Artificial Intelligence*, 4(3), p.e210064. <https://doi.org/10.1148/ryai.210064>
