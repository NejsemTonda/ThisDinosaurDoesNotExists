---
title: "Beamer Formatting Showcase"
subtitle: "Pandoc Markdown demo"
author: "Your Name"
date: \today
documentclass: beamer
theme: Madrid
colortheme: beaver
fonttheme: professionalfonts
aspectratio: 169
navigation: frame
---

# Overview

## Outline

This slide shows an automatically generated table of contents.

\tableofcontents


# Basic text

## Text styles


You can also get \textsc{Small Caps} via LaTeX.

Inline code like `x <- 1` and escaped characters: \#, \_, \%.

A forced line break here\\
and some more text on the next line.


## Paragraphs and inline emphasis

This is a normal paragraph. You can combine **bold** and *italic*,
and even **_both at once_**.

You can also use `\emph{LaTeX emphasis}` directly if you prefer.


## Blocks inside a frame

Headings *below* the slide level (here: level 3) become Beamer blocks.

### Regular block

This is a regular block created from a level-3 heading.

### Example block {.example}

This becomes an `exampleblock` in Beamer, useful for examples.

### Alert block {.alert}

This becomes an `alertblock` in Beamer, useful for warnings or key points.


# Lists

## Bullet and numbered lists

A simple bullet list:

- First bullet item
- Second bullet item
  - Nested bullet item
  - Another nested item
- Third bullet item

A numbered list:

1. First step
2. Second step
3. Third step


## Incremental list (per-slide)

Wrap a list in an `incremental` div to show items one by one,
even if you do *not* use the `-i` option:

::: incremental

- First point appears
- Then the second point
- Then the third point

:::

A non-incremental list inside the same document:

::: nonincremental

- All items appear at once
- Regardless of the `-i` option

:::


# Math and code

## Inline and display math

Inline math like $E = mc^2$ fits into text.

Display math:


## Code (fragile frame)

This frame is marked as , which is recommended when
using verbatim content.

Here is an indented code block:

    #include <stdio.h>

    int main(void) {
        printf("Hello, Beamer + Pandoc!\n");
        return 0;
    }

And another one:

    for i in range(5):
        print("Item", i)


# Layout

## Columns with text

Below, we use Pandoc's column layout. In Beamer, this becomes a `columns` environment.

:::::::::::::: {.columns align=center}
::: {.column width="45%"}
**Left column**

- Short bullet
- Another bullet
- A bit of text to pad it out

You can mix **bold**, *italic*, and math $a^2 + b^2 = c^2$ here.
:::
::: {.column width="55%"}
**Right column**

- This column is slightly wider
- You can add lists, text, or even images
- Columns are very useful for comparisons side by side
:::
::::::::::::::


## Image and caption

A simple figure using an example image from the `mwe` package
(loaded via `\usepackage{mwe}`):

![An example image from the mwe package](example-image){ width=0.6\textwidth }

If your TeX installation does not have the `mwe` package (and thus no `example-image`),
replace `example-image` with the path to one of your own images.


# Tables and footnotes

## Simple table

A basic pipe table:

| Item        | Description            | Value |
|-------------|------------------------|------:|
| Apples      | Number of apples       |     5 |
| Oranges     | Number of oranges      |     3 |
| Bananas     | Number of bananas      |     7 |


## Footnotes

You can add footnotes like this:

This sentence has a footnote marker.[^demo-footnote]

[^demo-footnote]: This is the content of the footnote. In Beamer, it appears at the bottom of the slide.


# Advanced frame attributes

## Standout frame

This frame uses the `.standout` class, which Beamer styles as a special "highlight" slide
(depending on your theme).

Use this kind of frame for big "Key takeaway" messages or section breaks.


## Long content with frame breaks 

This frame has the `.allowframebreaks` class, which allows Beamer to split it into multiple
slides if the content becomes too long.

- Item 1
- Item 2
- Item 3
- Item 4
- Item 5
- Item 6
- Item 7
- Item 8
- Item 9
- Item 10
- Item 11
- Item 12

(Depending on your theme and font size, this may or may not actually force a split,
but the option is set.)


# Speaker notes

## Slide with notes

This slide shows how to attach speaker notes that appear only in presenter mode
(Beamer notes / handouts).

Here is the content the audience sees:

- Visible bullet 1
- Visible bullet 2

::: notes

These are speaker notes.

- They will *not* appear on the slide itself.
- They will show up in handout / notes output depending on how you compile.

:::


# The End

## Thank you!

- Questions?
- Comments?
- Feedback?

You have now seen examples of:

- Text formatting and blocks
- Lists (including incremental lists)
- Math (inline and display)
- Code in a fragile frame
- Columns
- Images and captions
- Tables and footnotes
- Frame attributes (`.fragile`, `.standout`, `.allowframebreaks`)
- Speaker notes


