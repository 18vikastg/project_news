// Add smooth scrolling for anchor links
document.addEventListener('DOMContentLoaded', function() {
    function rememberSubmitButtonDefaults() {
        document.querySelectorAll('form button[type="submit"]').forEach(function (btn) {
            if (btn.dataset.defaultSubmitHtml === undefined) {
                btn.dataset.defaultSubmitHtml = btn.innerHTML;
            }
        });
    }

    rememberSubmitButtonDefaults();

    // BFCache / back button: the dashboard can be restored with the submit still disabled.
    window.addEventListener('pageshow', function () {
        rememberSubmitButtonDefaults();
        document.querySelectorAll('form button[type="submit"]').forEach(function (btn) {
            btn.disabled = false;
            if (btn.dataset.defaultSubmitHtml !== undefined) {
                btn.innerHTML = btn.dataset.defaultSubmitHtml;
            }
        });
    });

    // Add loading states to buttons (optional per-form message: data-submit-loading-html)
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function () {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (!submitBtn) {
                return;
            }
            if (submitBtn.dataset.defaultSubmitHtml === undefined) {
                submitBtn.dataset.defaultSubmitHtml = submitBtn.innerHTML;
            }
            const custom = this.getAttribute('data-submit-loading-html');
            submitBtn.innerHTML =
                custom ||
                '<i class="fas fa-spinner fa-spin"></i> Processing...';
            submitBtn.disabled = true;
        });
    });

    // Add animation to stats counters
    const animateCounters = () => {
        const counters = document.querySelectorAll('.stat-item h3');
        counters.forEach(counter => {
            if (!counter.classList.contains('animated')) {
                const target = counter.textContent;
                if (target.includes('%')) {
                    animateValue(counter, 0, parseInt(target), 2000);
                }
                counter.classList.add('animated');
            }
        });
    };

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start) + '%';
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Intersection Observer for counter animation
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounters();
            }
        });
    });

    const statsSection = document.querySelector('.stats-section');
    if (statsSection) {
        observer.observe(statsSection);
    }
});