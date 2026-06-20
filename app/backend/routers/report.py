from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter(prefix="/v1/report", tags=["report"])


@router.get("/ward/{ward_id}/pdf")
def ward_report(ward_id: str):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        text = f"UrbanCool AI ward report\nWard: {ward_id}\nInstall reportlab for PDF output.\n"
        return Response(text, media_type="text/plain")

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"UrbanCool AI Ward Report {ward_id}")
    c.drawString(72, 800, "UrbanCool AI Ward Heat Report")
    c.drawString(72, 775, f"Ward ID: {ward_id}")
    c.drawString(72, 750, "This report is generated from local open-data artifacts.")
    c.drawString(72, 725, "Use within-only ward rows for strict ward analysis.")
    c.showPage()
    c.save()
    return Response(buf.getvalue(), media_type="application/pdf", headers={"Content-Disposition": f"inline; filename=ward_{ward_id}_report.pdf"})

