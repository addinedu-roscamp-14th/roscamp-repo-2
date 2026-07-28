import io
import csv
import os
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash, Response, send_file
from sqlalchemy.exc import IntegrityError
from models import db, Admin, Service, Workspace, Reservation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

main_bp = Blueprint('main', __name__)

# --- 사용자 페이지 라우트 ---

@main_bp.route('/')
def intro():
    return render_template('intro.html')

@main_bp.route('/reserve')
def index():
    return render_template('index.html')

@main_bp.route('/login', methods=['GET', 'POST'])
def customer_login():
    if request.method == 'POST':
        # 고객 로그인 성공 처리
        session['customer_logged_in'] = True
        return redirect(url_for('main.intro'))
    return render_template('login.html')

@main_bp.route('/logout')
def customer_logout():
    # 모든 세션 완전 초기화
    session.clear()
    return redirect(url_for('main.intro'))

# --- API 라우트 (프론트엔드 연동) ---

@main_bp.route('/api/services', methods=['GET'])
def get_services():
    services = Service.query.all()
    return jsonify([{'id': s.id, 'name': s.name, 'duration': s.duration, 'color': s.color} for s in services])

@main_bp.route('/api/workspaces', methods=['GET'])
def get_workspaces():
    workspaces = Workspace.query.all()
    return jsonify([{'id': w.id, 'name': w.name} for w in workspaces])

@main_bp.route('/api/reservations', methods=['GET', 'POST'])
def handle_reservations():
    if request.method == 'GET':
        reservations = Reservation.query.all()
        res_list = []
        is_admin = session.get('admin_logged_in') # 관리자 로그인 여부 확인
        
        for r in reservations:
            # 관리자 로그인 시 실명 노출, 미로그인(또는 일반 고객) 시 "예약 완료"로 가림
            display_title = f"{r.customer_name} ({r.car_number})" if is_admin else "예약 완료"
            
            res_list.append({
                'id': r.id,
                'title': display_title,
                'start': r.start_time.isoformat(),
                'end': r.end_time.isoformat(),
                'resourceId': r.workspace_id,
                'color': r.services[0].color if r.services else '#3788d8',
                'extendedProps': {
                    'phone': r.customer_phone if is_admin else "",
                    'memo': r.memo if is_admin else "",
                    'services': [s.name for s in r.services]
                }
            })
        return jsonify(res_list)

    if request.method == 'POST':
        data = request.json
        try:
            start_time = datetime.fromisoformat(data['start_time'])
            # 타임존 정보 강제 제거
            start_time = start_time.replace(tzinfo=None)
            service_ids = data['service_ids']
            
            if 3 <= start_time.hour < 5:
                return jsonify({'success': False, 'message': '03:00 ~ 05:00은 시스템 점검으로 예약 불가 시간임.'}), 400
            
            if start_time < datetime.now():
                return jsonify({'success': False, 'message': '과거 시간은 예약 불가함.'}), 400

            services = Service.query.filter(Service.id.in_(service_ids)).all()
            if not services:
                return jsonify({'success': False, 'message': '선택한 서비스가 유효하지 않음.'}), 400

            total_duration = sum(s.duration for s in services)
            end_time = start_time + timedelta(minutes=total_duration)

            with db.session.begin_nested():
                busy_workspaces = db.session.query(Reservation.workspace_id).filter(
                    (Reservation.start_time < end_time) & (Reservation.end_time > start_time)
                ).subquery()

                available_workspace = Workspace.query.filter(
                    ~Workspace.id.in_(busy_workspaces)
                ).first()

                if not available_workspace:
                    return jsonify({'success': False, 'message': '선택한 시간에 모든 작업 공간이 사용 중임.'}), 400

                new_reservation = Reservation(
                    customer_name=data['customer_name'],
                    customer_phone=data['customer_phone'],
                    car_number=data['car_number'],
                    memo=data.get('memo', ''),
                    start_time=start_time,
                    end_time=end_time,
                    workspace_id=available_workspace.id
                )
                new_reservation.services.extend(services)
                db.session.add(new_reservation)
            
            db.session.commit()
            return jsonify({'success': True, 'message': '예약 확정됨.', 'reservation_id': new_reservation.id}), 201

        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'}), 500

@main_bp.route('/api/reservations/<int:res_id>', methods=['DELETE'])
def delete_reservation(res_id):
    reservation = Reservation.query.get_or_404(res_id)
    try:
        db.session.delete(reservation)
        db.session.commit()
        return jsonify({'success': True, 'message': '예약 취소 완료됨.'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'}), 500

@main_bp.route('/api/available-times', methods=['GET'])
def get_available_times():
    date_str = request.args.get('date')
    slots = []
    
    for hour in range(24):
        if hour == 3 or hour == 4:
            continue
            
        for minute in (0, 30):
            time_str = f"{hour:02d}:{minute:02d}"
            slots.append({'time': time_str, 'available': True})
            
    return jsonify({'slots': slots})

# --- 관리자 라우트 ---

@main_bp.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        admin = Admin.query.filter_by(username=username).first()
        if admin and admin.check_password(password):
            session['admin_logged_in'] = True
            return redirect(url_for('main.admin_dashboard'))
        flash('인증 실패함.')
    return render_template('admin/login.html')

@main_bp.route('/admin/logout')
def admin_logout():
    # 모든 세션 완전 초기화
    session.clear()
    return redirect(url_for('main.intro'))

@main_bp.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('main.admin_login'))
    
    today = datetime.now().date()
    reservations = Reservation.query.all()
    today_reservations = [r for r in reservations if r.start_time.date() == today]
    
    return render_template('admin/dashboard.html', today_count=len(today_reservations), reservations=reservations)

@main_bp.route('/admin/reservations')
def admin_reservations():
    if not session.get('admin_logged_in'):
        return redirect(url_for('main.admin_login'))
    reservations = Reservation.query.order_by(Reservation.start_time.desc()).all()
    return render_template('admin/reservations.html', reservations=reservations)

@main_bp.route('/admin/services', methods=['GET', 'POST'])
def admin_services():
    if not session.get('admin_logged_in'):
        return redirect(url_for('main.admin_login'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        duration = request.form.get('duration')
        color = request.form.get('color')
        new_service = Service(name=name, duration=int(duration), color=color)
        db.session.add(new_service)
        db.session.commit()
        flash('서비스 추가 완료됨.')
        return redirect(url_for('main.admin_services'))
        
    services = Service.query.all()
    return render_template('admin/services.html', services=services)

@main_bp.route('/admin/workspaces', methods=['GET', 'POST'])
def admin_workspaces():
    if not session.get('admin_logged_in'):
        return redirect(url_for('main.admin_login'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        new_workspace = Workspace(name=name)
        db.session.add(new_workspace)
        db.session.commit()
        flash('작업 공간 추가 완료됨.')
        return redirect(url_for('main.admin_workspaces'))
        
    workspaces = Workspace.query.all()
    return render_template('admin/workspaces.html', workspaces=workspaces)

@main_bp.route('/admin/export/csv')
def export_csv():
    if not session.get('admin_logged_in'):
        return redirect(url_for('main.admin_login'))
    
    reservations = Reservation.query.all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['예약번호', '고객명', '연락처', '차량번호', '시작시간', '종료시간', '작업공간', '서비스', '메모'])
    
    for r in reservations:
        services_str = ", ".join([s.name for s in r.services])
        writer.writerow([r.id, r.customer_name, r.customer_phone, r.car_number, 
                         r.start_time.strftime('%Y-%m-%d %H:%M'), 
                         r.end_time.strftime('%Y-%m-%d %H:%M'), 
                         r.workspace.name, services_str, r.memo])
    
    output.seek(0)
    return Response(
        output.getvalue().encode('utf-8-sig'),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=reservations.csv"}
    )

@main_bp.route('/api/reservation/<int:res_id>/pdf')
def generate_pdf(res_id):
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors

    reservation = Reservation.query.get_or_404(res_id)
    buffer = io.BytesIO()
    
    # 폰트 적용
    font_path = os.path.join(os.path.dirname(__file__), 'NanumGothic.ttf')
    try:
        pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
        current_font = 'NanumGothic'
    except:
        current_font = 'Helvetica'
    
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    # 스타일 정의
    title_style = ParagraphStyle(name='Title', fontName=current_font, fontSize=24, alignment=1, spaceAfter=20)
    heading_style = ParagraphStyle(name='Heading', fontName=current_font, fontSize=14, textColor=colors.HexColor('#1E3A8A'), spaceAfter=10, spaceBefore=20)
    
    elements.append(Paragraph("작업지시서 (Work Order)", title_style))
    
    # 정보 섹션
    elements.append(Paragraph("기본 정보", heading_style))
    info_data = [
        ['예약번호', str(reservation.id), '예약일시', reservation.start_time.strftime('%Y-%m-%d %H:%M')],
        ['고객명', reservation.customer_name, '차량번호', reservation.car_number],
        ['연락처', reservation.customer_phone, '작업공간', reservation.workspace.name]
    ]
    info_table = Table(info_data, colWidths=[80, 180, 80, 180])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, -1), current_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(info_table)
    
    # 작업 섹션
    elements.append(Paragraph("작업 항목", heading_style))
    service_data = [['현장 확인', '서비스 항목', '소요시간(분)']]
    for s in reservation.services:
        service_data.append(['[  ]', s.name, str(s.duration)])
        
    service_table = Table(service_data, colWidths=[80, 360, 80])
    service_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), current_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(service_table)
    
    # 메모 섹션
    elements.append(Paragraph("특이사항 및 메모", heading_style))
    memo_text = reservation.memo if reservation.memo else "특이사항 없음"
    memo_table = Table([[memo_text]], colWidths=[520], rowHeights=[120])
    memo_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), current_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(memo_table)
    
    doc.build(elements)
    buffer.seek(0)
    
    # PDF 인라인 열람 모드 적용 (as_attachment=False)
    return send_file(buffer, as_attachment=False, mimetype='application/pdf')

@main_bp.route('/inquiry')
def inquiry():
    return render_template('inquiry.html')