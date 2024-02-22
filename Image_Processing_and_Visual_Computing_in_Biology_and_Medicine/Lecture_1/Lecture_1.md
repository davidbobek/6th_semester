# Biomedical Imaging
- Strucural Imaging
    - X-ray
    - CT
    - MRI
    - Ultrasound
- Functional Imaging
    - PET
    - SPECT
    - fMRI


## Microscopy
- Histopathology: Study of tissues under microscope
- Staining: Dyeing the tissues to make them visible under microscope
- Reflectance Confocal Microscopy: 
    - Uses a laser to scan the tissue and create a black and white image
    - Used in dermatology
- UltraSound Microscopy
    - Uses high frequency sound waves to create an image
    - Used in imaging of internal organs (baby in womb)
- Medical Sonography
    - Uses ultrasound to visualize subcutaneous body structures
    - Used in obstetrics
- UltraSound Localization Microscopy
    - Uses ultrasound with microscopes to visualize cells
    - Used in cancer research
- Optical Coherence Tomography
    - Uses light to capture micrometer-resolution, 3D images from within optical scattering media
    - Used in ophthalmology (retina imaging)
- ColonoScopy
    - Uses a camera to visualize the colon
    - Used in colon cancer screening
    - The most invasive of all the imaging techniques
- Xray Microscopy
    - Uses X-rays to visualize cells
    - Safe to use on living cells and tissues
    - Cant pass through thick tissues and bones
- MamoGraphy
    - Uses X-rays to visualize the breast
    - Used in breast cancer screening
- Digital Subtraction Angiography
    - Uses X-rays to visualize blood vessels
    - Used in angiography
- Computed Tomography (CT)
    - Uses X-rays to visualize the body in 3D
    - Used in cancer diagnosis
    - Rotating X-ray machine
- Magnetic Resonance Imaging (MRI)
    - Uses magnetic fields to visualize the body in 3D
    - CANT go with metals 
    - Hydrogen atoms are used to align the magnetic field
    - Can show much more detail than CT due to  
-  Emisson Tomography
    - Uses radioactive tracers to visualize the body in 3D
    - Used in cancer diagnosis
    - PET
        - Positron Emission Tomography
        - Combined with CT/MRI for conology imaging
        - Better quality images 
    - SPECT
        - Single Photon Emission Computed Tomography
        - Bone scans/ Heart scans
        - Lower quality images
    
- PhotoAcoustic Tomography
    - light heats up the tissue and creates sound waves
    - 3D images of the body are created
    - Used in cancer diagnosis
    - Latest Technology

## What is a PACS
- PACS
    - Picture Archiving and Communication System
    - Medical imaging technology used for storing, retrieving, presenting, and sharing images produced by various medical hardware modalities
    - Used in radiology, cardiology, pathology, and ophthalmology

## Radioloigst responsibilities
- Detecting and diagnosing diseases
- Reporting the findings to the referring physician

## Human Vision
- Cone: Color vision
- Rod: Night vision

- Colorblindness is cauase by the absence of cones in the eye
- 3 types of cones: Red, Green, Blue
- L - Cones: Long wavelength: Red
- M - Cones: Medium wavelength: Green
- S - Cones: Short wavelength: Blue

## AI for Medical Imaging
- Needs big data
- Transfer learning
- Data augmentation

- Benefits
    - Faster diagnosis
    - Save lives

- Challenge
    - Different types of modalities
        (CT, MRI, X-ray, Ultrasound)
    - Lack of data 

## Medical Imaging Usages
- Early detection of diseases
- Measurements 
- Biomarkers characterization
- Disease progression
- Lesion Segmentation

## What is a digital image
- Creating a discrete representation of an image from a continuous one
- Pixels: Smallest unit of a digital image
- Resolution: The level of detail in an image

- Feature function: A function that maps a feature vector to a real number
- Quantization: Process of mapping a continuous range of values to a finite range of values

- Think about an image as a function
    - f(x,y) = Pixel value at (x,y)
    - f(x,y,z) = Voxel value at (x,y,z)
    - f(x,y,z,t) = Time series of 3D images

## CT Image Pixel Values
- Each pixel is assigned a number (Hounsfield Unit)
- Hounsfield Unit: A quantitative scale for describing radiodensity

```bash
HU = 1000 * (μ - μw) / μw
    - μ: Linear attenuation coefficient of the material
    - μw: Linear attenuation coefficient of water
```
the higher the HU, the denser the material

## MRI Contrast
- Basis for imaging: Radio waves emitted by hydrogen atoms
- T1 Weighted Image
    - Fat is bright
    - Water is dark
- T2 Weighted Image
    - Fat is dark
    - Water is bright

## Imagining Planes
- Axial: Horizontal
- Coronal: Vertical
- Sagittal: Vertical but from the side


## Image Processing Formats
- DICOM: Digital Imaging and Communications in Medicine
    - Standard for handling, storing, printing, and transmitting information in medical imaging
    - Key:Value pairs

- NIfTI: Neuroimaging Informatics Technology Initiative
    - Standard for storing and sharing neuroimaging data
    - 3D and 4D images
    - Header file and image file
    - NII
    - Many DICOM together

## Supervised Learning
- Classification
- Regression

## Unsupservised Learning
- Clustering
- Dimensionality Reduction
- Anomaly Detection
- t-SNE
- PCA
- UMAP

## Process of Image Processing
- Classification
- Detection
- Segmentation

## Semantic Segmentation vs Instance Segmentation
- Semantic Segmentation
    - Assigns a class to each pixel
- Instance Segmentation
    - Assigns a class to each object

## Image Processing Techniques
- Image Enhancement: Improving the quality of the image
- Image Restoration: Repairing the image
- Image Compression: Reducing the size of the image
- Image Analysis: Extracting information from images
- Image Synthesis: Creating new images from existing ones

## Radiomics
- Extracting quantitative features from medical images 
- Preprocessing
    - Contrast enhancement

- ROI: Region of Interest
    - Tumor
    - Lesion
    - Organ

- Histogram Equalization
    - Improves the contrast of the image
    - Spreads out the intensity values
    - Cumulative Distribution Function

- Quantization
    - Enables the segmentation of the image
    - Reduces the number of gray levels

