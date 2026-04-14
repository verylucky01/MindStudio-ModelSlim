感谢您贡献的Pull Request！
在提交之前，请务必阅读 [CONTRIBUTING.md](../docs/zh/appendix/CONTRIBUTING.md)。

Thanks for sending a pull request!
BEFORE SUBMITTING, PLEASE READ [CONTRIBUTING.md](../docs/zh/appendix/CONTRIBUTING.md).

## PR描述 (What this PR does / why we need it?)

- 请明确说明您提交PR的变更内容。本部分旨在概述所做的变更，以及此PR是如何解决该问题的。请尽可能地提供有助于评审人员更高效、更快速完成检视审查的实用说明。
- 请说明为何需要这些更改，例如具体的使用场景或bug描述。
- 关联issue号（如果有）。

- Please clarify what changes you are proposing. The purpose of this section is to outline the changes and how this PR fixes the issue. If possible, please consider writing useful notes for better and faster reviews in your PR.
- Please clarify why the changes are needed. For instance, the use case and bug description.
- Related issue number (if any)


## 面向用户的变更 (Does this PR introduce _any_ user-facing change)?
- 请注意，这里指的是**任何**面向用户的变更，包括但不限于API、用户界面或其他使用方式上的变更。

- Note that it means *any* user-facing change including all aspects such as API, interface or other behavior changes.



## 功能验证 (How was this patch tested?)

请确认CI已通过增量及存量的单元测试用例。
如果本次测试方式与常规单元测试不同，请详细说明您的测试步骤(最好提供完整的可复现的操作路径及关键截图)，以便Committer能够快速复现验证，也便于后续的维护。
如果未添加测试，请说明未添加的原因，以及为何难添加测试。

- [_] 功能自验
- [_] 本地自验截图(涉及个人标识符等敏感信息请注意脱敏)
- [_] 新增/变更内容是否已新增/适配UT测试用例看护

CI passed with new added/existing test.
If it was tested in a way different from regular unit tests, please clarify how you tested step by step, ideally copy and paste-able, so that other reviewers can test and check, and descendants can verify in the future.
If tests were not added, please describe why they were not added and/or why it was difficult to add.

- [_] Self-verification of the feature.
- [_] Screenshot of local self-verification (please anonymize any sensitive information such as personal identifiers)
- [_] Have new or modified unit test (UT) cases been added or adapted to cover the newly added or changed content?
