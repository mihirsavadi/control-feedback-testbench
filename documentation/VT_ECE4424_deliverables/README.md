# Virginia Tech ECE4424 Project Specification

*NOTE: The documents in this directory pertain to my deliverables for the 'ECE 4424 Machine Learning' class I took in
the Spring of 2022 at Virginia Tech, taught by [Prof. Debswapna
Bhattacharya](https://cs.vt.edu/people/faculty/deb-bhattacharya.html). The efforts involving this repository, the
Control-Feedback Testbench, and the Real-Time Convergent Feedback Loop all grew out of this class' final project
deliverable.*

The course project is meant for students to (1) gain experience implementing machine learning models; and (2) try
machine learning on problems that interest them. You are encouraged to try out interesting applications of machine
learning in various domains such as vision, NLP, speech, computational biology, etc. The project must be done
individually in this semester (i.e., no double counting).

The first deliverable (15% of course grade) is a project proposal that is due on March 4. The project proposal should
identify the problem, outline your preliminary approach, and propose the metrics for evaluation. It should also discuss
a proposed plan containing a breakdown of various tasks and important project milestones. These milestones should be a
prediction for planning purposes, but you are not obligated to adhere to them precisely. Your proposal should list at
least three recent, relevant papers you will read and understand as background. The project proposal must be written
using the following guidelines:

standard 8.5" x 11" page size 11 point or higher font, except text that is part of an image Times New Roman font for all
text, Cambria Math font for equations, Symbol font for non-alphabetic characters (it is recommended that equations and
symbols be inserted as an image) 1" margins on all sides, no text inside 1" margins (no header, footer, name, or page
number) No less than single-spacing (approximately 6 lines per inch) Do not use line spacing options such as "exactly 11
point", that are less than single spaced The project proposal is required to be between 2 - 3 pages in PDF file format
only to be submitted electronically via Canvas. The page limit includes all references, citations, charts, figures, and
images.

## Project Proposal Grading (75 points)

The course staff will follow the National Science Foundation (NSF)-style evaluation metrics to review and score your
project proposal as Excellent (5 points), Very Good (4 points), Good (3 points), Fair (2 points), and Poor (1 point).
Three reviews will be sought each reviewing and scoring the proposals, and the (sum of points * 5) will be your final
score for the project proposal.

The final deliverable (20% of course grade) is a project report that is due on May 4 (i.e., on the last day of classes).
The final project report should describe the project outcomes in a self-contained manner. Your final project report is
required to be between 5 - 6 pages by using the
[CVPR_template](http://cvpr2020.thecvf.com/sites/default/files/2019-09/cvpr2020AuthorKit.zip), structured like a paper
from a computer vision conference, to be submitted electronically via Canvas. Please use this template so we can fairly
judge all student projects without worrying about altered font sizes, margins, etc. The submitted PDF can link to
supplementary materials including but not limited to code, open access software package via GitHub, project webpage,
videos, and other supplementary material. The final PDF project report should completely address all of the points in
the rubric described below.

## Project Report Grading Rubric (100 points)

Note: We have adapted the following list for our rubric based on [questions for evaluating research
projects](https://www.darpa.mil/work-with-us/heilmeier-catechism) proposed by a former DARPA director George H.
Heilmeier and recently used by Dhruv Batra for teaching (Deep Learning course at Georgia
Tech)[https://www.cc.gatech.edu/classes/AY2022/cs7643_fall/].

Introduction / Background / Motivation:
- (10 points) What did you try to do? What problem did you try to solve? Articulate your objectives using absolutely no
  jargon.
- (5 points) How is it done today, and what are the limits of current practice?
- (5 points) Who cares? If you are successful, what difference will it make?

Approach:
- (10 points) What did you do exactly? How did you solve the problem? Why did you think it would be successful? Is
  anything new in your approach?
- (5 points) What problems did you anticipate? What problems did you encounter? Did the very first thing you tried work?

Experiments and Results:
- (10 points) How did you measure success? What experiments were used? What were the results, both quantitative and
  qualitative? Did you succeed? Did you fail? Why?

Availability:
- (5 points) Is your code available? Did you use open-source license to release your code?
- (10 points) How do you plan to disseminate your method? Are the findings available via freely accessible project
  website and/or GitHub?

Reproducibility:
- (10 points) How can others reproduce your results? Are training, validation, and test data freely provided?
- (5 points) Are model parameters fully reproducible?

In addition, 25 more points will be distributed based on:
- (10 points) Appropriate use of figures / tables / visualizations. Are the ideas presented with appropriate
  illustration? Are the results presented clearly; are the important differences illustrated?
- (5 points) Overall clarity. Is the manuscript self-contained? Can a peer who has also taken Machine Learning
  understand all of the points addressed above? Is sufficient detail provided?
- (10 points) Finally, points will be distributed based on your understanding of how your project relates to Machine
  Learning. Here are some questions to think about:
    - What was the structure of your problem? How did the structure of your model reflect the structure of your problem?
    - What parts of your model had learned parameters (e.g., convolution layers) and what parts did not (e.g.,
      post-processing classifier probabilities into decisions)?
    - What representations of input and output did the model expect? How was the data pre/post-processed?
    - What was the loss function?
    - Did the model overfit? How well did the approach generalize?
    - What hyperparameters did the model have? How were they chosen? How did they affect performance? What optimizer was
      used?
    - What Machine Learning framework did you use?
    - What existing code or models did you start with and what did those starting points provide?

An **intermediate deliverable** (not graded, but submission is mandatory) is a midway project progress check due on
April 25. The midway project progress should be in the same format as the project proposal and discuss the progress made
and any changes to the original plan. The midway project progress should also contain an updated breakdown of the tasks
and the final project milestones. The final project report may not be graded if the midway project progress check is not
submitted. If you are struggling to make progress in the project, this would be an ideal time to seek help.