/*!
 * Start Bootstrap - Clean Blog v6.0.9 (https://startbootstrap.com/theme/clean-blog)
 * Copyright 2013-2023 Start Bootstrap
 * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-clean-blog/blob/master/LICENSE)
 */
window.addEventListener("DOMContentLoaded", () => {
    let scrollPos = 0;
    const mainNav = document.getElementById("mainNav");
    const headerHeight = mainNav.clientHeight;
    window.addEventListener("scroll", function () {
        const currentTop = document.body.getBoundingClientRect().top * -1;
        if (currentTop < scrollPos) {
            // Scrolling Up
            if (currentTop > 0 && mainNav.classList.contains("is-fixed")) {
                mainNav.classList.add("is-visible");
            } else {
                mainNav.classList.remove("is-visible", "is-fixed");
            }
        } else {
            // Scrolling Down
            mainNav.classList.add("is-visible");
            if (
                currentTop > headerHeight &&
                !mainNav.classList.contains("is-fixed")
            ) {
                mainNav.classList.add("is-fixed");
            }
        }
        scrollPos = currentTop;
    });
});

document.addEventListener("DOMContentLoaded", function () {
    const targetField = document.getElementById("target");
    const formFields = [
        "Alcohol",
        "Malic_Acid",
        "Ash",
        "Ash_Alcanity",
        "Magnesium",
        "Total_Phenols",
        "Flavanoids",
        "Nonflavanoid_Phenols",
        "Proanthocyanins",
        "Color_Intensity",
        "Hue",
        "OD280",
        "Proline",
    ];

    function updateFormFields() {
        const selectedTarget = targetField.value;

        formFields.forEach((field) => {
            const fieldElement = document.getElementById(`${field}`);
            if (!selectedTarget || selectedTarget === field) {
                fieldElement.classList.add("d-none");
            } else {
                fieldElement.classList.remove("d-none");
            }
        });
    }

    // Initial setup: hide all fields except the selected target
    updateFormFields();

    // Event listener for target field change
    targetField.addEventListener("change", updateFormFields);
});