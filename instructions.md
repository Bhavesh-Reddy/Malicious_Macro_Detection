
## Step 1: Accept the Repository Invitation

1. Check your GitHub notifications or email.
2. Accept the invitation to join the repository.
3. Open the repository in your browser.

---

## Step 2: Open Your Assigned Issue

1. Click **Issues**.
2. Open the issue assigned to you.
3. Follow the instruction given during start of the session.
---

## Step 3: Create a New Branch

1. Click the branch selector at the top-left (it will usually show **main**).
2. Click **View all branches** (if shown).
3. Click **New branch**.
4. Name your branch using this format:

```
yourname-taskname
```

Example:

```
john-CPP_Data (4545-4565)
```

5. Click **Create branch**.

> **Do not work directly on the `main` branch.**

---

## Step 4: Navigate to the Correct Folder

Example:

```
datasets/
    python/
```

or

```
datasets/
    cpp/
```

depending on your assigned task.

---

## Step 5: Upload Your JSONL File

1. Click **Add file**.
2. Select **Upload files**.
3. Drag and drop your `.jsonl` file.
4. Wait for the upload to finish.

---

## Step 6: Commit the File

At the bottom of the page:

Commit message:

```
Added CPP Data
```

Choose:

* **Commit directly to your branch** (`yourname-taskname`)

Click:

**Commit changes**

---

## Step 7: Create a Pull Request

After committing, GitHub usually displays:

```
Compare & pull request
```

Click it.

If it doesn't appear:

1. Click **Pull requests**.
2. Click **New pull request**.
3. Set:

   * Base branch: `main`
   * Compare branch: your branch

Click:

**Create pull request**

---

## Step 8: Pull Request Description

Include:

```
Completed Issue #<issue number>

Uploaded:
- dataset_name.jsonl

Output Format:
JSONL

Closes #<issue number>
```

Example:

```
Completed Issue #12

Uploaded:
python_dataset_001.jsonl

Output Format:
JSONL

Closes #12
```

Then click:

**Create pull request**

---

## Step 9: Wait for Review

The project manager will review your submission.

If changes are requested:

1. Open your branch.
2. Update the `.jsonl` file.
3. Commit the changes to the **same branch**.

The existing Pull Request will update automatically.

---

## Step 10: Completion

Once the Pull Request is approved and merged:

* Your task is complete.
* The assigned issue will be closed automatically.
* Your dataset becomes part of the main repository.

---

# Important Rules

* Work only on your own branch.
* Do not upload files to the `main` branch.
* Upload only `.jsonl` files unless instructed otherwise.
* Validate your JSONL before submitting.
* Follow the assigned folder structure.
* If you have questions, ask in the issue comments before submitting.
