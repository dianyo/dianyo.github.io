(function () {
  'use strict';
  if (!window.katex) return;

  // Explicit math containers keep DNA strings and interactive output untouched.
  document.querySelectorAll('.k-equation, .k-math').forEach(function (element) {
    const displayMode = element.classList.contains('k-equation');
    const source = element.textContent.trim().replace(/^\\[[(]\s*|\s*\\[\])]$/g, '');
    try {
      window.katex.render(source, element, {
        displayMode: displayMode,
        output: 'htmlAndMathml',
        throwOnError: true,
        trust: false,
        strict: 'error'
      });
      element.dataset.mathRendered = 'true';
      if (displayMode) {
        element.tabIndex = 0;
        element.setAttribute('role', 'region');
        element.setAttribute('aria-label', 'Equation; scroll horizontally if needed');
      }
    } catch (error) {
      // Leave readable LaTeX in place if an equation cannot be rendered.
      element.dataset.mathError = 'true';
      console.error('Unable to render article equation:', error.message);
    }
  });
}());
