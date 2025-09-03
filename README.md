# Medical Imaging Processing System

A comprehensive Python-based medical imaging processing framework that implements advanced Digital Fourier Transform (DFT) techniques, frequency domain filtering, and selective image acquisition patterns for medical image analysis.

## Overview

This project provides a robust set of tools for medical image processing with a focus on:
- Frequency domain analysis using Digital Fourier Transform
- Advanced filtering techniques (Ideal, Gaussian, Butterworth, Ring filters)
- Selective image acquisition patterns (Cartesian, Radial, Elliptical, Band patterns)
- Signal-to-noise ratio analysis
- Medical image enhancement and noise reduction

## Project Structure

```
FinalProjectMI/
├── src/
│   ├── project/                    # Core modules
│   │   ├── Utilities.py           # Image processing utilities and DFT operations
│   │   ├── ImageSynthesisNoise.py # Frequency domain filtering algorithms  
│   │   └── SelectiveImageAcquisition.py # Acquisition pattern generation
│   └── interface/                  # Demo applications and results
│       ├── ImageAcquisitionDriver.py    # Acquisition pattern demonstrations
│       ├── ImageNoiseDriver.py          # Filtering demonstrations
│       └── images/                      # Sample medical images
├── test/                          # Comprehensive test suite
│   ├── test_Utilities.py         # Tests for utility functions
│   ├── test_ImageSynthesisNoise.py # Tests for filtering algorithms
│   ├── test_SelectiveImageAcquisition.py # Tests for acquisition patterns
│   └── resources/                 # Test data and expected outputs
├── requirements.txt               # Python dependencies
└── README.md                     # Project documentation
```

## Features

### Core Image Processing Utilities (`Utilities.py`)
- **Image I/O Operations**: Load/save images and matrices in various formats
- **Digital Fourier Transform**: Forward and inverse DFT operations
- **Image Normalization**: Pixel value normalization for consistent processing
- **Mask Application**: Apply frequency domain masks to DFT representations
- **Signal-to-Noise Ratio**: Calculate SNR for image quality assessment
- **Post-processing**: Convert processed images to displayable formats

### Frequency Domain Filtering (`ImageSynthesisNoise.py`)
- **Ideal Filters**: Sharp cutoff lowpass and highpass filters
- **Gaussian Filters**: Smooth rolloff lowpass and highpass filters  
- **Butterworth Filters**: Configurable order lowpass and highpass filters
- **Ring Filters**: Bandpass and bandstop ring-shaped filters
- **Customizable Parameters**: Cutoff frequencies, filter orders, and ring dimensions

### Selective Image Acquisition (`SelectiveImageAcquisition.py`)
- **Cartesian Patterns**: Horizontal line acquisition with configurable density
- **Circular Patterns**: Circular region acquisition with adjustable radius
- **Elliptical Patterns**: Elliptical acquisition regions with rotation support
- **Band Patterns**: Rectangular bands with arbitrary orientation
- **Radial Patterns**: Spoke-like radial acquisition patterns
- **Spiral Patterns**: Spiral trajectory acquisition (framework provided)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sanrasta/FinalProjectMI.git
   cd FinalProjectMI
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
- `opencv-python ~= 4.4.0.46` - Computer vision and image processing
- `numpy ~= 1.19.4` - Numerical computing and array operations
- `setuptools ~= 46.4.0` - Package management utilities

## Usage

### Basic Image Processing Workflow

```python
import src.project.Utilities as util
import src.project.ImageSynthesisNoise as noise
import src.project.SelectiveImageAcquisition as aqc

# Load and preprocess medical image
image = util.loadImage("path/to/medical_image.jpg")
normalized_image = util.normalizeImage(image)

# Transform to frequency domain
dft_image = util.getDFT(normalized_image)

# Create acquisition pattern
height, width = image.shape
mask = aqc.radialPattern((height, width), ray_count=20)

# Apply mask and reconstruct
masked_dft = util.applyMask(dft_image, mask)
reconstructed = util.getImage(masked_dft)
final_image = util.post_process_image(reconstructed)

# Save result
util.saveImage("output.jpg", final_image)
```

### Frequency Domain Filtering

```python
# Create Gaussian lowpass filter
gaussian_filter = noise.gaussianLowpassFilter((512, 512), cutoff=50)

# Create Butterworth highpass filter  
butterworth_filter = noise.butterworthHighpassFilter((512, 512), cutoff=30, order=2)

# Apply filter to DFT representation
filtered_dft = util.applyMask(dft_image, gaussian_filter)
filtered_image = util.getImage(filtered_dft)
```

### Running Demonstrations

The project includes demonstration scripts that showcase various processing techniques:

```bash
# Navigate to interface directory
cd src/interface

# Run acquisition pattern demonstrations
python ImageAcquisitionDriver.py

# Run filtering demonstrations  
python ImageNoiseDriver.py
```

## Testing

The project includes a comprehensive test suite covering all major functionality:

```bash
# Run all tests
python -m pytest test/

# Run specific test modules
python -m unittest test.test_Utilities
python -m unittest test.test_ImageSynthesisNoise  
python -m unittest test.test_SelectiveImageAcquisition
```

### Test Coverage
- **Utilities Tests**: Image I/O, DFT operations, mask application
- **Filtering Tests**: All filter types with various parameters
- **Acquisition Tests**: All acquisition patterns with edge cases

## Applications

This framework is particularly useful for:

- **MRI Image Processing**: Simulating k-space acquisition patterns
- **CT Reconstruction**: Implementing sparse acquisition techniques
- **Image Denoising**: Applying frequency domain filters for noise reduction
- **Research Applications**: Studying the effects of different acquisition strategies
- **Medical Image Enhancement**: Improving image quality through filtering
- **Educational Purposes**: Learning frequency domain image processing concepts

## API Reference

### Core Functions

#### `Utilities.py`
- `loadImage(path)` - Load grayscale image from file
- `getDFT(image)` - Compute centered 2D DFT
- `getImage(dft)` - Inverse DFT to reconstruct image
- `applyMask(dft, mask)` - Apply acquisition mask to DFT
- `signalToNoise(img1, img2)` - Calculate SNR between images
- `post_process_image(image)` - Normalize image for display

#### `ImageSynthesisNoise.py`
- `idealLowpassFilter(size, cutoff)` - Create ideal lowpass filter
- `gaussianLowpassFilter(size, cutoff)` - Create Gaussian lowpass filter
- `butterworthLowpassFilter(size, cutoff, order)` - Create Butterworth lowpass filter
- `ringLowpassFilter(size, cutoff, thickness)` - Create ring filter

#### `SelectiveImageAcquisition.py`
- `cartesianPattern(size, percent)` - Generate Cartesian acquisition pattern
- `radialPattern(size, ray_count)` - Generate radial acquisition pattern
- `ellipsePattern(size, major, minor, angle)` - Generate elliptical pattern
- `bandPattern(size, width, length, angle)` - Generate oriented band pattern

## Contributing

This project follows academic software development standards:

1. **Code Style**: Follow PEP 8 Python style guidelines
2. **Testing**: Maintain comprehensive test coverage for all new features
3. **Documentation**: Update README and inline documentation for changes
4. **Compatibility**: Ensure compatibility with specified dependency versions

## Technical Notes

### Performance Considerations
- DFT operations use NumPy's optimized FFT implementation
- Large images may require significant memory for frequency domain processing
- Filter generation is computationally intensive for large image sizes

### Medical Image Compatibility
- Supports standard medical imaging formats (DICOM, PNG, JPEG)
- Handles grayscale images optimally
- Maintains image intensity relationships critical for medical analysis

### Numerical Precision
- Uses double precision floating point for DFT operations
- Implements proper scaling for forward/inverse transforms
- Handles complex number operations correctly

## License

This project is developed for academic and research purposes. Please refer to the original repository for licensing information.

## Authors

- **@sanrasta** - Project lead and core development
- **@Justin-WR-West** - Additional development and testing
- **@jrpower3** - Development contributions

## Acknowledgments

This project was developed as part of advanced medical imaging coursework, focusing on practical applications of frequency domain processing techniques in medical image analysis.
