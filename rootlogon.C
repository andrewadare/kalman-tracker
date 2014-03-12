{
  if (gSystem->Getenv("TMPDIR")) 
    gSystem->SetBuildDir(gSystem->Getenv("TMPDIR"));
}