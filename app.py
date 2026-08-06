import subprocess
import sys
from flask import Flask, render_template, request, jsonify, session
from config import Config
from models import db, Admin, Service, Workspace, Reservation, DeletedReservation
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# 초기 설정
with app.app_context():
    db.create_all()
    # 최초 관리자 계정 생성
    if not Admin.query.filter_by(username='core').first():
        admin = Admin(username='core')
        admin.set_password('1')
        db.session.add(admin)
    # 기본 작업 공간 생성
    if Workspace.query.count() == 0:
        for i in range(1, 5): db.session.add(Workspace(name=f'베이 {i}'))
    db.session.commit()

from routes import main_bp
app.register_blueprint(main_bp)

# --- 관리자 인증 유틸리티 ---
def is_admin():
    return session.get('admin_logged_in') == True

# --- 예약 관련 API (검색, 취소, 삭제) ---
@app.route('/api/reservation/search', methods=['POST'])
def search_reservation():
    data = request.get_json()
    query = Reservation.query
    if data.get('name'): query = query.filter_by(customer_name=data['name'])
    elif data.get('car'): query = query.filter_by(car_number=data['car'])
    elif data.get('phone'): query = query.filter_by(customer_phone=data['phone'])
    else: return jsonify({'success': False}), 400
    
    res = query.order_by(Reservation.start_time.desc()).first()
    if not res: return jsonify({'success': False})
    return jsonify({'success': True, 'data': {
        'id': f"RES-{res.id}", 'raw_id': res.id, 'name': res.customer_name,
        'car': res.car_number, 'phone': res.customer_phone,
        'date': res.start_time.strftime('%Y.%m.%d %H:%M')
    }})

@app.route('/api/reservation/cancel/<int:res_id>', methods=['POST'])
def cancel_reservation(res_id):
    res = Reservation.query.get_or_404(res_id)
    db.session.delete(res)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin/reservation/delete/<int:res_id>', methods=['POST'])
def admin_delete_reservation(res_id):
    if not is_admin(): return jsonify({'success': False}), 403
    res = Reservation.query.get_or_404(res_id)
    db.session.delete(res)
    db.session.commit()
    return jsonify({'success': True})

# --- 시간 선택 통합 로직 API ---
@app.route('/api/available-times')
def get_available_times():
    target_date = request.args.get('date')
    # 해당 날짜의 예약된 시간들 조회
    reservations = Reservation.query.filter(Reservation.start_time.like(f'{target_date}%')).all()
    reserved_times = [r.start_time.strftime('%H:%M') for r in reservations]
    
    # 09:00 ~ 18:00, 30분 단위 슬롯
    slots = []
    for h in range(9, 18):
        for m in ['00', '30']:
            t = f'{h:02}:{m}'
            slots.append({'time': t, 'available': t not in reserved_times})
    return jsonify({'slots': slots})

# --- 서비스 및 작업공간 관리 API ---
@app.route('/admin/service/update/<int:s_id>', methods=['POST'])
def update_service(s_id):
    if not is_admin(): return jsonify({'success': False}), 403
    s = Service.query.get_or_404(s_id)
    data = request.json
    s.name = data.get('name', s.name)
    s.price = data.get('price', s.price)
    s.duration = data.get('duration', s.duration)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin/workspace/create', methods=['POST'])
def create_workspace():
    if not is_admin(): return jsonify({'success': False}), 403
    name = request.json.get('name')
    if name:
        db.session.add(Workspace(name=name))
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

@app.route('/admin/workspace/update/<int:ws_id>', methods=['POST'])
def update_workspace(ws_id):
    if not is_admin(): return jsonify({'success': False}), 403
    ws = Workspace.query.get_or_404(ws_id)
    ws.name = request.json.get('name')
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin/workspace/delete/<int:ws_id>', methods=['POST'])
def delete_workspace(ws_id):
    if not is_admin(): return jsonify({'success': False}), 403
    ws = Workspace.query.get_or_404(ws_id)
    db.session.delete(ws)
    db.session.commit()
    return jsonify({'success': True})

if __name__ == '__main__':
    fastapi_process = None
    try:
        # main.py 서브 프로세스 실행
        fastapi_process = subprocess.Popen([sys.executable, "main.py"])
        # Flask 서버 실행
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        # 종료 시 서브 프로세스 종료 처리
        if fastapi_process:
            fastapi_process.terminate()
            fastapi_process.wait()