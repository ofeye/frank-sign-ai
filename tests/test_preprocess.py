"""Tests for preprocessing module."""

import pytest
from pathlib import Path
from PIL import Image
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from franksign.data.preprocess import preprocess_images, _iter_images


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def temp_dirs():
    """Create temporary input/output directories."""
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    yield Path(input_dir), Path(output_dir)
    # Cleanup
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture
def sample_images(temp_dirs):
    """Create sample images in temp input directory."""
    input_dir, output_dir = temp_dirs
    
    # Create 3 sample images
    for i, name in enumerate(["test1.jpg", "test2.png", "test3.jpeg"]):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(input_dir / name)
    
    # Create nested directory with image
    nested = input_dir / "subdir"
    nested.mkdir()
    img = Image.new("RGB", (200, 150), color=(100, 100, 100))
    img.save(nested / "nested.jpg")
    
    return input_dir, output_dir


# ============================================================
# TESTS FOR preprocess_images
# ============================================================

class TestPreprocessImages:
    """Tests for preprocess_images function."""
    
    def test_processes_all_images(self, sample_images):
        """All images should be processed."""
        input_dir, output_dir = sample_images
        processed, skipped = preprocess_images(input_dir, output_dir)
        
        assert processed == 4
        assert skipped == 0
    
    def test_creates_output_directory(self, temp_dirs):
        """Output directory is created if it doesn't exist."""
        input_dir, output_dir = temp_dirs
        new_output = output_dir / "nested_new_output"  # Use output_dir, not input_dir
        
        # Create one image
        img = Image.new("RGB", (100, 100))
        img.save(input_dir / "test.jpg")
        
        preprocess_images(input_dir, new_output)
        
        assert new_output.exists()
        assert (new_output / "test.jpg").exists()
    
    def test_resizes_images(self, sample_images):
        """Images should be resized to target size."""
        input_dir, output_dir = sample_images
        target_size = (64, 128)  # height, width
        
        preprocess_images(input_dir, output_dir, image_size=target_size)
        
        output_image = Image.open(output_dir / "test1.jpg")
        assert output_image.size == (128, 64)  # PIL uses (width, height)
    
    def test_skips_existing_without_overwrite(self, sample_images):
        """Existing files are skipped when overwrite=False."""
        input_dir, output_dir = sample_images
        
        # First run
        preprocess_images(input_dir, output_dir)
        
        # Second run without overwrite
        processed, skipped = preprocess_images(input_dir, output_dir, overwrite=False)
        
        assert processed == 0
        assert skipped == 4
    
    def test_overwrites_with_flag(self, sample_images):
        """Files are overwritten when overwrite=True."""
        input_dir, output_dir = sample_images
        
        # First run
        preprocess_images(input_dir, output_dir)
        
        # Second run with overwrite
        processed, skipped = preprocess_images(input_dir, output_dir, overwrite=True)
        
        assert processed == 4
        assert skipped == 0
    
    def test_preserves_directory_structure(self, sample_images):
        """Nested directory structure is preserved."""
        input_dir, output_dir = sample_images
        
        preprocess_images(input_dir, output_dir)
        
        assert (output_dir / "subdir" / "nested.jpg").exists()
    
    def test_converts_to_rgb(self, temp_dirs):
        """RGBA images are converted to RGB."""
        input_dir, output_dir = temp_dirs
        
        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img.save(input_dir / "rgba.png")
        
        preprocess_images(input_dir, output_dir)
        
        output_image = Image.open(output_dir / "rgba.png")
        assert output_image.mode == "RGB"
    
    def test_empty_directory(self, temp_dirs):
        """Empty input directory returns zeros."""
        input_dir, output_dir = temp_dirs
        
        processed, skipped = preprocess_images(input_dir, output_dir)
        
        assert processed == 0
        assert skipped == 0


# ============================================================
# TESTS FOR _iter_images
# ============================================================

class TestIterImages:
    """Tests for _iter_images helper function."""
    
    def test_finds_jpg_images(self, temp_dirs):
        """Finds .jpg files."""
        input_dir, _ = temp_dirs
        Image.new("RGB", (10, 10)).save(input_dir / "test.jpg")
        
        images = list(_iter_images(input_dir))
        assert len(images) == 1
        assert images[0].suffix == ".jpg"
    
    def test_finds_multiple_formats(self, temp_dirs):
        """Finds jpg, png, jpeg, bmp, tif files."""
        input_dir, _ = temp_dirs
        
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            Image.new("RGB", (10, 10)).save(input_dir / f"test{ext}")
        
        images = list(_iter_images(input_dir))
        assert len(images) == 5
    
    def test_ignores_non_image_files(self, temp_dirs):
        """Non-image files are ignored."""
        input_dir, _ = temp_dirs
        
        (input_dir / "readme.txt").write_text("hello")
        (input_dir / "data.csv").write_text("a,b,c")
        Image.new("RGB", (10, 10)).save(input_dir / "image.jpg")
        
        images = list(_iter_images(input_dir))
        assert len(images) == 1
    
    def test_case_insensitive(self, temp_dirs):
        """Extension matching is case-insensitive."""
        input_dir, _ = temp_dirs
        
        Image.new("RGB", (10, 10)).save(input_dir / "test.JPG")
        Image.new("RGB", (10, 10)).save(input_dir / "test2.Png")
        
        images = list(_iter_images(input_dir))
        assert len(images) == 2
    
    def test_finds_nested_images(self, temp_dirs):
        """Finds images in nested directories."""
        input_dir, _ = temp_dirs
        
        nested = input_dir / "a" / "b" / "c"
        nested.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(nested / "deep.jpg")
        
        images = list(_iter_images(input_dir))
        assert len(images) == 1
        assert "deep.jpg" in str(images[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
