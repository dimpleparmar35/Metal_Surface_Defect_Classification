document.addEventListener('DOMContentLoaded', () => {
    // Select all sections that need to fade in
    const sections = document.querySelectorAll('.fade-in-section');

    // Create an Intersection Observer
    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            // When the section comes into view
            if (entry.isIntersecting) {
                // Add the visible class to trigger CSS transition
                entry.target.classList.add('is-visible');
                // Optional: Stop observing once it has faded in
                observer.unobserve(entry.target);
            }
        });
    }, {
        // Trigger when 15% of the section is visible
        threshold: 0.15,
        rootMargin: '0px 0px -50px 0px'
    });

    // Start observing each section
    sections.forEach(section => {
        observer.observe(section);
    });
});
