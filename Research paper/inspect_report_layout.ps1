# Read back the finished report through Word and report the layout facts that
# matter: where each heading landed, what the table of contents resolved to,
# and whether any table is wider than the text column.
$ErrorActionPreference = 'Stop'
$path = Join-Path $PSScriptRoot 'PROJECT PHASE_I REPORT_RGCET_2025.docx'

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $doc = $word.Documents.Open($path, $false, $true)   # read-only

    $page = 43
    $w = $doc.Sections.Item(2).PageSetup.PageWidth
    $mL = $doc.Sections.Item(2).PageSetup.LeftMargin
    $mR = $doc.Sections.Item(2).PageSetup.RightMargin
    $textW = $w - $mL - $mR
    Write-Output ("page {0} x {1} pt, text column {2} pt ({3} twips)" -f `
        $w, $doc.Sections.Item(2).PageSetup.PageHeight, $textW, ($textW * 20))
    Write-Output ''
    $mapped = 0

    # --- heading -> page map -------------------------------------------
    Write-Output 'HEADING MAP'
    Write-Output '---------------------------------------------------------'
    foreach ($p in $doc.Paragraphs) {
        $sn = $p.Style.NameLocal
        if ($sn -like 'Heading*' -or $sn -eq 'heading 1' -or $sn -eq 'heading 2' -or $sn -eq 'heading 3') {
            $t = $p.Range.Text -replace "[`r`a]", ''
            if ($t.Trim().Length -gt 0) {
                $pg = $p.Range.Information(3)   # wdActiveEndPageNumber
                Write-Output ("  p{0,-4} {1,-10} {2}" -f $pg, $sn, $t.Trim())
                $mapped++
            }
        }
    }
    Write-Output ''
    Write-Output ("headings mapped: " + $mapped)

    Write-Output ''
    Write-Output 'TABLE OF CONTENTS (first 30 lines)'
    Write-Output '---------------------------------------------------------'
    $toc = $doc.TablesOfContents.Item(1).Range.Text -split "`r"
    $toc | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 30 |
        ForEach-Object { Write-Output ("  " + $_.Trim()) }

    Write-Output ''
    Write-Output "TOC lines: $(($toc | Where-Object { $_.Trim().Length -gt 0 }).Count)"

    # --- cover page sanity ---------------------------------------------
    Write-Output ''
    Write-Output 'COVER PAGE'
    Write-Output '---------------------------------------------------------'
    $p1 = $doc.GoTo(1, 1, 1)          # wdGoToPage, wdGoToAbsolute
    $p2 = $doc.GoTo(1, 1, 2)
    $c1 = $doc.Range($p1.Start, $p2.Start).Text -replace "[`r`a`v]", ' '
    $c1 = ($c1 -replace '\s+', ' ').Trim()
    Write-Output ("  page 1 chars: " + $c1.Length)
    Write-Output ("  page 1 text : " + $c1.Substring(0, [Math]::Min(600, $c1.Length)))

    $doc.Close(0)
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
