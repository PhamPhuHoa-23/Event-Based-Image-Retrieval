document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Handle lazy loading of images
    const lazyLoadImages = function() {
        const imgElements = document.querySelectorAll('img[data-src]');
        const windowHeight = window.innerHeight;
        
        imgElements.forEach(function(img) {
            const rect = img.getBoundingClientRect();
            // Check if image is in viewport
            if (rect.top >= 0 && rect.top <= windowHeight) {
                const src = img.getAttribute('data-src');
                // Create a new image to check if it loads successfully
                const newImg = new Image();
                newImg.onload = function() {
                    img.src = src;
                    img.removeAttribute('data-src');
                };
                newImg.onerror = function() {
                    // If image fails to load, use placeholder
                    img.classList.add('placeholder-img');
                    img.removeAttribute('data-src');
                    
                    // Add icon to placeholder
                    const imgParent = img.parentElement;
                    if (imgParent && !imgParent.querySelector('.placeholder-icon')) {
                        const icon = document.createElement('i');
                        icon.className = 'fas fa-image placeholder-icon';
                        imgParent.appendChild(icon);
                    }
                };
                newImg.src = src;
            }
        });
    };

    // Initial check for images
    lazyLoadImages();
    
    // Check for images on scroll
    window.addEventListener('scroll', function() {
        lazyLoadImages();
    });
    
    // Handle image zoom on click
    document.querySelectorAll('.result-image, .article-image').forEach(function(img) {
        img.addEventListener('click', function(e) {
            if (!this.classList.contains('placeholder-img')) {
                const modal = document.createElement('div');
                modal.className = 'modal fade';
                modal.id = 'imageModal';
                modal.tabIndex = '-1';
                modal.setAttribute('aria-hidden', 'true');
                
                const modalDialog = document.createElement('div');
                modalDialog.className = 'modal-dialog modal-dialog-centered modal-lg';
                
                const modalContent = document.createElement('div');
                modalContent.className = 'modal-content';
                
                const modalBody = document.createElement('div');
                modalBody.className = 'modal-body text-center p-0';
                
                const fullImg = document.createElement('img');
                fullImg.src = this.src;
                fullImg.className = 'img-fluid';
                fullImg.style.maxHeight = '80vh';
                
                const closeBtn = document.createElement('button');
                closeBtn.className = 'btn-close position-absolute top-0 end-0 m-2';
                closeBtn.setAttribute('data-bs-dismiss', 'modal');
                closeBtn.setAttribute('aria-label', 'Close');
                
                modalBody.appendChild(fullImg);
                modalBody.appendChild(closeBtn);
                modalContent.appendChild(modalBody);
                modalDialog.appendChild(modalContent);
                modal.appendChild(modalDialog);
                
                document.body.appendChild(modal);
                
                // Initialize and show the modal
                const bsModal = new bootstrap.Modal(modal);
                bsModal.show();
                
                // Remove modal from DOM when hidden
                modal.addEventListener('hidden.bs.modal', function() {
                    document.body.removeChild(modal);
                });
            }
        });
    });
}); 