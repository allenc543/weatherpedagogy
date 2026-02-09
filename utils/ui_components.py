"""Shared UI components: concept boxes, quizzes, navigation, page templates."""
import streamlit as st


def chapter_header(number, title, part=None):
    """Render a chapter header with part label."""
    if part:
        st.caption(f"Part {part}")
    st.title(f"Chapter {number}: {title}")
    st.divider()


def concept_box(title, content):
    """Render a highlighted concept/theory box."""
    st.markdown(f"""
<div style="background-color: #EBF5FB; padding: 20px; border-radius: 10px; border-left: 5px solid #2E86C1; margin: 10px 0;">
<h4 style="color: #2E86C1; margin-top: 0;">{title}</h4>
<p style="color: #1B4F72;">{content}</p>
</div>
""", unsafe_allow_html=True)


def formula_box(title, formula, explanation=""):
    """Render a formula with explanation."""
    st.markdown(f"**{title}**")
    st.latex(formula)
    if explanation:
        st.caption(explanation)


def insight_box(text):
    """Render a key insight callout."""
    st.info(f"**Key Insight:** {text}")


def warning_box(text):
    """Render a warning/common mistake box."""
    st.warning(f"**Common Mistake:** {text}")


def code_example(code, language="python"):
    """Render a collapsible code example."""
    with st.expander("Show Code"):
        st.code(code, language=language)


def quiz(question, options, correct_idx, explanation="", key="quiz"):
    """Render a multiple-choice quiz question. Returns True if answered correctly."""
    st.subheader("Quick Quiz")
    answer = st.radio(question, options, key=key, index=None)
    if answer is not None:
        if options.index(answer) == correct_idx:
            st.success("Correct!")
            if explanation:
                st.caption(explanation)
            return True
        else:
            st.error(f"Not quite. The correct answer is: **{options[correct_idx]}**")
            if explanation:
                st.caption(explanation)
            return False
    return None


def takeaways(points):
    """Render key takeaways as a list."""
    st.subheader("Key Takeaways")
    for p in points:
        st.markdown(f"- {p}")


def navigation(prev_label=None, next_label=None, prev_page=None, next_page=None):
    """Render prev/next navigation buttons."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if prev_label:
            st.page_link(f"pages/{prev_page}", label=f"← {prev_label}")
    with col3:
        if next_label:
            st.page_link(f"pages/{next_page}", label=f"{next_label} →")


def progress_metric(label, value, delta=None):
    """Render a metric card."""
    st.metric(label, value, delta)
