# Preliminary report checklist

The implementation supplies calculations and illustrations. It does not replace
the team's engineering explanation, assigned hand calculation, or physical test.

| Requirement | Output or remaining team work |
|---|---|
| Introduction and purpose of computation | Explain the design comparison and why computation is useful |
| Contributions and percent effort | Team writes actual contributions |
| Theory and assumptions | Explain equations and assumptions using `method.md` |
| Assigned practice-truss verification | Team solves assigned problem by hand and compares with code |
| Two labeled design diagrams | `figures/geometry.*` and force diagrams |
| Member number, length, T/C, force at maximum load | `members.csv`, `members.tex`, `analysis.md` |
| Buckling strength and uncertainty for compressed members | Member table and `buckling_capacity.*` |
| First member(s) to buckle and maximum load uncertainty | `summary.json` and `analysis.md` |
| Cost and maximum-load/cost ratio | Summary and comparison figure |
| Specification compliance | Complete check list and `cut_plan.csv` |
| Design comparison and improvements | Team interprets results and search history |
| AI use and lessons learned | Describe assistance, independent checks, and understanding |
| MATLAB input for first design | Rename `baseline/design.mat` to the required group filename |
| Analysis code | Submit the Python source as instructed by course staff |
| Individual peer assessment | Each student completes the course form |

For the current report draft, check these theory points before submission:

1. `C` has J rows and M columns. Each member is a column with two connected joints.
2. Counting equations is necessary but insufficient for stability; geometry matters.
3. Describe computational analysis as faster/repeatable. An "exponential" speedup
   is a mathematical claim that has not been established by this project.
4. Use "best geometry found within the selected topology" when describing this
   optimizer's result. There is no proof of the best possible truss topology.

Do not confuse the independent analytical tests in this repository with the
specific practice problem assigned by the course. That problem has not been
included here. Record the team's own hand solution and agreement in the report.

The preliminary report, code, and first-design `.mat` are due November 13, 2026
at 11:59 p.m. according to the supplied October 14 draft manual. Verify any
later course announcements. The report sections are introduction, method and
analysis, results, and discussion/conclusion, with a title page and relevant
appendices. Include units, figure/table references, and equation definitions.
