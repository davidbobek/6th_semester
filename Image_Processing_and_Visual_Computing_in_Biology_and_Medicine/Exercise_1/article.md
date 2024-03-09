## Title: Python-Powered Precision: CT Image Segmentation for Lung Extraction in DICOM Format

This article is going to guide you through the process of extracting lung images from CT scans using Python. We will be using the DICOM format for this purpose.

## Introduction
The technology stack which is going to be used in this article is as follows:
- Python
- Pydicom
- Numpy
- Matplotlib
- Scikit-image

The objective of this guide is to provide a step-by-step approach to extract lung images from CT scans using Python. The guide is going to be divided into the following sections:
- Understanding the DICOM format
- Initial visualization of the CT scan
- Preprocessing the CT scan
- Thresholding the CT scan based on the Otsu method
- Extracting the lung images from the CT scan
- Improving the lung extraction using morphological operations
- 3D visualization of the extracted lung images

## Understanding the DICOM format
DICOM stands for Digital Imaging and Communications in Medicine. In the current era, it is the standard for handling, storing and managing medical imaging information. Its key features include:
- Being able to store a wide range of information
- Being able to store images in a series
- Having the ability to store metadata in the form of key-value pairs

## Initial visualization of the CT scan
The first step in the process is to visualize the CT scan. We will be using the Pydicom library to read the DICOM files and visualize the CT scan. The following code snippet is going to be used for this purpose:

```python
scans_path = "LungCT-Diagnostic Pre-Surgery Contrast Enhanced" 
list_of_scans = os.listdir(scans_path)

# for figuring out the controls lets experiment with slice 122 of slice 2
scan_num = 2
scan_path = os.path.join(scans_path, list_of_scans[scan_num])
list_of_slices = os.listdir(scan_path)
slice_num = 42
slice_path = os.path.join(scan_path, list_of_slices[slice_num])

# read in the full path to the file as ds
ds=dicom.read_file(slice_path) # you may have to use pydicom instead of dicom 

# rawimg is the pixel array of the image
rawimg= ds.pixel_array
plt.imshow(rawimg, cmap='viridis')
plt.show()
```
Output:

This code snippet is going to read the DICOM file and visualize the CT scan using the matplotlib library. The output is going to be a 2D image of the CT scan. 
On this image we can see the lungs and the surrounding tissues. The next step is tgoing to show us an ordered sequence of CT scans in order for us to better understand the progression of the 3D image.

```python
def multi_visualization(ds,list_of_slices,scan_path,plotting=True):
    volCT = np.zeros((ds.pixel_array.shape[0], ds.pixel_array.shape[1], len(list_of_slices)))
    print(volCT.shape)


    all_dcom_files = [dicom.read_file(os.path.join(scan_path, slice)) for slice in list_of_slices]
    # In theory len(all_dcom_files) == list_of_slices
    assert len(all_dcom_files) == len(list_of_slices)


    for dcom in all_dcom_files:
        slice_num = int(dcom.InstanceNumber) - 1
        volCT[:,:,slice_num] = dcom.pixel_array

    # Order the slices by InstanceNumber
    sorted_dcom_files = sorted(all_dcom_files, key=lambda x: int(x.InstanceNumber))
    sorted_slices = [dcom.pixel_array for dcom in sorted_dcom_files]

    for i, dcom in enumerate(sorted_dcom_files):
        slice_num = int(dcom.InstanceNumber) - 1
        volCT[:,:,slice_num] = sorted_slices[i]
        

    if plotting==True:
        # Display 25 slices of the 3D volume
        fig, ax = plt.subplots(5, 5, figsize=[15,15])
        for i in range(5):
            for j in range(5):
                ax[i, j].imshow(volCT[:,:,(i*5+j)], cmap='viridis')
                ax[i, j].axis('off')

        plt.show()
    return volCT
```
Output:

This code snippet creates 25 ordered slices of the 3D volume of the CT scan. The output is a 5x5 grid of 2D images of the CT scan. We as humans can clearly see the progression of the 3D image and the area where the lungs are located. The next step is to preprocess the CT scan in order to extract the lung images.

## Preprocessing the CT scan
In order to extract the lung images from the CT scan, we need to preprocess the CT scan. This steps converts the data into a format which is going to be easier to work with. This format is called float32. After conversion to float32 we are going to normalize the data from 0 to 1. This allows us to work with the data in a more efficient manner. The following code snippet is going to be used for this purpose:

```python
def preprocess_CT(volCT):
    volume_float32 = volCT.astype(np.float32)

    # Normalize to range [0, 1]
    volume_float32 /= np.max(volume_float32)

    # Output the min, max, and dtype of the voxels before and after conversion
    print("Before conversion - Min:", np.min(volCT), "Max:", np.max(volCT), "Dtype:", volCT.dtype)
    print("After conversion - Min:", np.min(volume_float32), "Max:", np.max(volume_float32), "Dtype:", volume_float32.dtype)

    # Check that the conversion worked
    assert volume_float32.dtype == np.float32

    # Check that the normalization worked
    assert np.min(volume_float32) == 0
    assert np.max(volume_float32) == 1
    return volume_float32
```

A set of assertions are used to check that the conversion and normalization worked. The output is going to be the min, max and dtype of the voxels before and after conversion. The next step is to threshold the CT scan based on the Otsu method.

## Thresholding the CT scan based on the Otsu method
The Otsu method is a thresholding method which is used to separate the background from the foreground in an image. It is based on the assumption that the image contains two classes of pixels. We are going to create a color histogram where the x-axis is the pixel value and the y-axis is the frequency of the pixel value.  

```python
    plt.figure(figsize=[10,5])
    plt.hist(volume_float32.flatten(), bins=1000)
    plt.axvline(x=threshold_otsu(volume_float32), color='r', linestyle='--')
    plt.title('Histogram of pixel values in the 3D volume')
    plt.xlim([0, 0.3])
    plt.show()
```
On this plot we can see 2 main peaks. The first peak is the background and the second peak is the foreground. In order for us to properly threshold the CT scan we need to find the value which separates the background from the foreground. This value is going to be the threshold. 
The threshold is going to be found using the Otsu method. The Otsu method works on the principle of minimizing the intra-class variance and maximizing the inter-class variance. 

```python
    def otsu_thresholding(volume_float32,plotting=True):
    # Find the Otsu's threshold    
    thresh = threshold_otsu(volume_float32)

    if plotting==True:
        plt.figure(figsize=[10,5])
        plt.hist(volume_float32.flatten(), bins=1000)
        plt.axvline(x=threshold_otsu(volume_float32), color='r', linestyle='--')
        plt.title('Histogram of pixel values in the 3D volume')
        plt.xlim([0, 0.3])
        plt.show()
        
    return thresh
```
Output:

On this plot we can see the threshold value which separates the background from the foreground. The next step is to extract the lung images from the CT scan. And apply morphological operations to improve the lung extraction. The binary mask places all the pixels which are above the threshold in the foreground and all the pixels which are below the threshold in the background. 

```python
    def binary_mask(volume_float32, thresh,plotting=True):
        # Apply the threshold
        binary_volume = volume_float32 > thresh

        # Visualize the result
        if plotting==True:
            plt.figure(figsize=[10,10])
            plt.imshow(binary_volume[:,:,20])
            plt.show()
            
        return binary_volume
```

The outcome of this step is a binary mask of the CT scan on which we humans can clearly see the lungs. The next step is to apply morphological operations to improve the lung extraction. 


## Extracting the lung images from the CT scan
We are going to work with the premise that lungs are of a different color than the tissue which surrounds them. We can also see that lungs and the empty space around the tissue have the same color. This is due to the reason that lungs are filled with air. We are going to use this information to improve the lung extraction

```python

from skimage.measure import label
def lung_separation(volume_float32, thresh, plotting=True):
    vol = volume_float32
    thresh = threshold_otsu(vol)

    # Make a binary volume mask of anything below the threshold
    volBW = vol < thresh

    # Find connected components in the binary mask
    labeled_vol, num_features = label(volBW, connectivity=1, return_num=True)

    # Iterate through each slice
    for s in range(labeled_vol.shape[0]):
        # Find all connected components in the slice
        connected_components, component_count = label(labeled_vol[s], connectivity=1, return_num=True)
        
        # For each connected component in the slice
        for c in range(1, component_count + 1):  # Start from 1 as 0 is background
            # Check if the connected component touches the edge of the image
            if np.any(connected_components == c) and (np.any(connected_components[0, :] == c) or 
                                                    np.any(connected_components[-1, :] == c) or 
                                                    np.any(connected_components[:, 0] == c) or 
                                                    np.any(connected_components[:, -1] == c)):
                # Set all pixels of the connected component in volBW to 0
                volBW[s][connected_components == c] = 0
                
    # Visualize the result
    if plotting==True:
        _, ax = plt.subplots(5,5, figsize=[15, 15])
        for i in range(5):
            for j in range(5):
                ax[i, j].imshow(volBW[:,:,i*5+j], cmap='viridis')
                ax[i, j].axis('off')
        plt.show()
    return volBW

```

This code snippet creates a natural progression of 25 CT-scan which have clearly separated lungs from the surrounding tissue. This has been achieved by trying to find the connected components in the binary mask allowing us to group the pixels which are connected to each other. The objective of removing the surrounding tissue was successful. The achieved outcome was successful however we can still see some noise in the images. The next step will involve a volume filtering in order to remove the noise from the images.

## Improving the lung extraction by using volume filtering
The volume filtering works on the premise of removing particles which are smaller than a certain threshold. The main assumption is that the particles which are smaller than the threshold are noise. This can be achieved by this approach:

```python
def volume_filterin(volBW, plotting=True):
    
    smoothed_vol = np.copy(volBW)
    V = 100

    # Iterate through each slice
    for s in range(smoothed_vol.shape[0]):
        # Find all connected components in the slice
        connected_components, component_count = label(smoothed_vol[s], connectivity=1, return_num=True)
        
        # For each connected component in the slice
        for c in range(1, component_count + 1):  # Start from 1 as 0 is background
            # Check if the connected component is smaller than V
            if np.sum(connected_components == c) < V:
                # Set all pixels of the connected component in smoothed_vol to 0
                smoothed_vol[s][connected_components == c] = 0

    # Visualize the result
    if plotting==True:
        _, ax = plt.subplots(5,5, figsize=[15, 15])
        for i in range(5):
            for j in range(5):
                ax[i, j].imshow(smoothed_vol[:,:,i*5+j], cmap='viridis')
                ax[i, j].axis('off')

        plt.show()
    return smoothed_vol
```

The outcome of this step has cleared the images of the noise and allowing us to be more precise in the lung extraction. The next step is to make the lungs more enclosed by using binary closing.

## Enclosing the lungs by using binary closing
The binary closing works on the premise of closing the holes in the binary mask. The outcome of this step shall make the lungs more rounder and more smooth around the edges. This can be achieved by this approach:

```python
# Smooth out the edges of the lungs
from skimage.morphology import binary_closing

def applying_binary_closing(smoothed_vol,plotting=True):
    # Iterate through each slice
    for s in range(smoothed_vol.shape[0]):
        # Perform binary closing on the slice
        smoothed_vol[s] = binary_closing(smoothed_vol[s])
        
    # Visualize the result
    if plotting==True:
        _, ax = plt.subplots(5,5, figsize=[15, 15])
        for i in range(5):
            for j in range(5):
                ax[i, j].imshow(smoothed_vol[:,:,i*5+j], cmap='viridis')
                ax[i, j].axis('off')
        plt.show()
    return smoothed_vol
```

The outcome of this step makes each of the individual slices more rounder and more appealing to the human eye. The next step is to visualize the 3D volume of the extracted lung images.

## 3D visualization of the extracted lung images
The whole process of extracting the lung images from the CT scan has been successful. The next step is to visualize the 3D volume of the extracted lung images. We are going to use the marching cubes algorithm to create a 3D mesh of the extracted lung images. The graphical representation of the 3D mesh is going to be created using the matplotlib library. Marching cubes is an algorithm which creates a 3D surface from a 3D volume. 

```python
from skimage.measure import marching_cubes

def rending_of_the_lungs_3D(smoothed_vol):
    # Extract the mesh
    verts, faces, _, _ = marching_cubes(smoothed_vol, level=0.5)

    # Visualize the mesh
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: verts[faces] gives the vertices of each triangle
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='viridis', lw=1)
    plt.show()
    return verts, faces
```

The outcome of this step is a 3D mesh of the extracted lung images. On our plot we can see both of the lungs separated from the surrounding tissue. This is the final step of the process. The outcome of this process is a 3D mesh of the extracted lung images. 

## Conclusion
This article has provided a step-by-step approach to extract lung images from CT scans using Python. The process has been successful and the outcome is a 3D mesh of the extracted lung images. This process can be used in the medical field to extract lung images from CT scans and can be used in the MLOps industry as input for the machine learning models


