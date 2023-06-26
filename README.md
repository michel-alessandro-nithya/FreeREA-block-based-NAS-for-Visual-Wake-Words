# FreeREA-block-based-NAS-for-Visual-Wake-Words

Deep Neural Networks (DNNs) have become ubiquitous across various domains, including the emerging field of Internet of Things (IoT). These devices have several limitations when it comes to memory and computing capability. Therefore, DNNs for tiny devices requires a balanced trade-off between performances and its footprint. Moreover, designing suitable networks with high performances may require huge computational resources and/or big human experience. Our goal was to implement a tiny DNN that can fit and run on edge devices, while respecting hardware constraints in terms of number of parameters and FLOPs. To reach our goal, we carried out an hardware-aware block-based Neural Architecture Search (NAS) with training free metrics. We trained our network on Visual Wake Words Dataset (VWW), a common vision microcontroller use-case, which involves the identification of the presence or absence of a person in an image. We reached a 90.32% accuracy on VWW test set for very low resolution images (128x128).

We kindly invite you to run surveillance_application.py on a Windows sufficiently updated OS after having installed opencv-python and pygame.

You can install them by running

`pip install opencv-python`
`pip install pygame`.
